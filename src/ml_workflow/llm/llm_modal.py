import json
from typing import Any

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import modal
import time

llm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "torch",
        "transformers[torch,vision]",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "bitsandbytes",
        "huggingface-hub==0.36.0",
        "ipython",
        "fastapi[standard]"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file("llm.py", remote_path="/root/llm.py")
    .add_local_file("prompts.py", remote_path="/root/prompts.py")
)

from prompts import BASE_PROMPT, EVIL_PROMPT, QUESTION_PROMPT
secrets = [modal.Secret.from_name("hf")]
HF_CACHE = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App(name="testing-llm-modal", 
                image=llm_image, 
                secrets=secrets,
                volumes={
                    "/root/.cache/huggingface": HF_CACHE,
                }
                )

@app.cls(gpu = 'H200', min_containers = 0, max_containers = 1, scaledown_window = 1000)
class LLM:
    @modal.enter()
    def initialize_model(self):
        # Device detection
        self.dtype = torch.bfloat16
        self.max_new_tokens = 700  # Changed from 2048 to match your prompt requirement
        model_name = 'medgemma-27b'
        base_prompt = BASE_PROMPT
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_name == "medgemma-4b":
            print(f"Loading model ({model_name})")
            model_id = "google/medgemma-4b-it"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map="auto",
            )

        elif model_name == 'medgemma-27b':
            print(f"Loading model ({model_name})")
            model_id = "google/medgemma-27b-text-it"
            
            # Load tokenizer
            self.processor = AutoTokenizer.from_pretrained(model_id)
            
            # Verify chat template exists
            if not hasattr(self.processor, 'chat_template') or self.processor.chat_template is None:
                print("WARNING: No chat template found, setting default")
                self.processor.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}assistant: "
            
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU name: {torch.cuda.get_device_name(0)}")
                print(f"Initial GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
            
            # Load model - NO QUANTIZATION for A100-80GB
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            print(f"Model dtype: {self.model.dtype}")
            print(f"Model device: {next(self.model.parameters()).device}")
            if torch.cuda.is_available():
                print(f"GPU memory after load: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
            
        elif model_name == 'none':
            self.model = None
        else:
            raise ValueError(f"Model {model_name} not found")

        # Format base prompt: strip leading/trailing whitespace but preserve structure
        if base_prompt is not None:
            self.base_prompt = ' '.join(base_prompt.split())
        else:
            self.base_prompt = ""

    def build_prompt(self, predictions: dict, metadata: dict) -> str:
        """
        Combine the base prompt with model predictions and metadata.
        predictions: dictionary of predicted conditions and their confidence scores
        metadata: dictionary of metadata
        """
    
        summary = []
        for pred, conf in predictions.items():
            summary.append(f"{pred} ({conf*100:.1f}%)")
        summary_str = ", ".join(summary)

        prompt = f"{self.base_prompt}\n\nINPUT DATA: {summary_str}.\n"

        if 'user_input' in metadata:
            prompt += f"User input: {metadata['user_input']}\n"

        if 'cv_analysis' in metadata:
            cv = metadata['cv_analysis']
            area = str(cv.get('area', 'N/A'))
            color_profile = cv.get('color_profile', {})
            parts = [f"area={area}"]
            if color_profile:
                redness = color_profile.get('redness_index', 'N/A')
                parts.append(f"redness={redness}")
            prompt += f"Latest CV Analysis: {', '.join(parts)}.\n"

        if 'history' in metadata and metadata['history']:
            prompt += "\nHistorical Data:\n"
            for date, data in sorted(metadata['history'].items()):
                prompt += f"Date: {date}.\n"
                if 'cv_analysis' in data:
                    cv = data['cv_analysis']
                    area = str(cv.get('area', 'N/A'))
                    color_profile = cv.get('color_profile', {})
                    parts = [f"area={area}"]
                    if color_profile:
                        redness = color_profile.get('redness_index', 'N/A')
                        parts.append(f"redness={redness}")
                    prompt += f"  CV Analysis: {', '.join(parts)}.\n"
                if 'text_summary' in data:
                    prompt += f"  Text Summary: {data['text_summary']}.\n"
        
        print(f"\n=== BUILT PROMPT ===\n{prompt[:500]}...\n")
        return prompt
        
    @torch.inference_mode()
    def get_input_ids(self, prompt: str) -> torch.Tensor:
        """Get input IDs for the model."""
        return self.processor(text=prompt, return_tensors="pt").to(self.model.device)

    @torch.inference_mode()
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate text output from the model given a prepared prompt."""
        print(f"\n=== GENERATION START ===")
        
        # MedGemma requires chat template formatting
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template (CRITICAL for MedGemma)
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(formatted_prompt)
        
        print(f"Formatted prompt length: {len(formatted_prompt)} chars")
        
        # Tokenization timing
        start_time = time.time()
        inputs = self.processor(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.model.device)
        tokenize_time = time.time() - start_time
        
        input_length = inputs['input_ids'].shape[1]
        print(f"Tokenization: {tokenize_time:.3f}s")
        print(f"Input tokens: {input_length}")
        print(f"Model device: {next(self.model.parameters()).device}")
        
        # GPU memory before generation
        if torch.cuda.is_available():
            print(f"GPU memory before gen: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
        
        # Create stopping criteria for "thought" marker
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class ThoughtStoppingCriteria(StoppingCriteria):
            """Stop generation when 'thought' token is encountered."""
            def __init__(self, tokenizer, prompt_length):
                self.tokenizer = tokenizer
                self.prompt_length = prompt_length
                # Tokenize "thought" to get its token ID(s)
                self.thought_tokens = tokenizer.encode("thought", add_special_tokens=False)
                self.thought_token = self.thought_tokens[0] if self.thought_tokens else None
                
            def __call__(self, input_ids, scores, **kwargs):
                # Check if the last generated token is "thought"
                if self.thought_token is not None:
                    # Only check tokens after the prompt
                    generated_ids = input_ids[0][self.prompt_length:]
                    if len(generated_ids) > 0 and generated_ids[-1] == self.thought_token:
                        return True
                return False
        
        stopping_criteria = StoppingCriteriaList([
            ThoughtStoppingCriteria(self.processor, input_length)
        ])
        
        # Generation with proper stopping criteria
        start_time = time.time()
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False if temperature > 0 else False,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.processor.pad_token_id,
            eos_token_id=self.processor.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        
        generation_time = time.time() - start_time
        output_length = outputs.shape[1] - input_length
        
        print(f"Generation: {generation_time:.3f}s")
        print(f"Output tokens: {output_length}")
        print(f"Tokens/sec: {output_length/generation_time:.2f}")
        
        # GPU memory after generation
        if torch.cuda.is_available():
            print(f"GPU memory after gen: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
        
        # Decode only new tokens
        start_time = time.time()
        generated_ids = outputs[0][input_length:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        decode_time = time.time() - start_time
        
        # Extract only content outside <thought> tags
        text = self._extract_user_response(text)
        
        print(f"Decoding: {decode_time:.3f}s")
        print(f"Total time: {tokenize_time + generation_time + decode_time:.3f}s")
        print(f"=== GENERATION END ===\n")
        
        return text
    
    def _extract_user_response(self, text: str) -> str:
        """Extract only the user-facing response, removing thought markers and their content."""
        import re
        
        # Handle both XML-style tags and plain "thought" markers
        # Remove everything inside <thought>...</thought> tags (including the tags)
        text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Handle plain "thought" marker at the end (common pattern: "...text\nthought\nreasoning...")
        if 'thought' in text.lower():
            # Find "thought" as a standalone word (not part of another word like "thoughtful")
            thought_match = re.search(r'\bthought\b', text, re.IGNORECASE)
            if thought_match:
                # Cut everything from "thought" onwards
                thought_pos = thought_match.start()
                text = text[:thought_pos]
        
        # Also handle unclosed <thought> tags
        if '<thought>' in text.lower():
            thought_pos = text.lower().find('<thought>')
            text = text[:thought_pos]
        
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace 3+ newlines with 2
        text = text.strip()
        
        return text

    @modal.fastapi_endpoint(method="POST", docs=True)
    def explain(self, json_data: dict) -> str:
        """
        Full convenience method: build the prompt + run the model.
        predictions: dictionary of predicted conditions and their confidence scores
        metadata: dictionary of metadata
        """
        predictions = json_data['predictions']
        metadata = json_data['metadata']
        prompt = self.build_prompt(predictions, metadata)
        answer = self.generate(prompt, 0.3)
        return answer
    
    @modal.fastapi_endpoint(method="POST", docs=True)
    def ask_followup(self, json_data: dict) -> dict:
        """
        Ask a follow-up question based on the initial analysis.
        
        Expected json_data format:
        {
            "initial_answer": str,
            "question": str,
            "conversation_history": list (optional) - list of previous questions (max 5)
        }
        """
        initial_answer = json_data['initial_answer']
        question = json_data['question']
        conversation_history = json_data.get('conversation_history', [])
        
        # Keep only last 5 questions and add the current one
        conversation_history = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
        conversation_history.append(question)
        
        # Build conversation context
        history_text = ""
        if len(conversation_history) > 1:
            history_text = "\n\nPrevious questions in this conversation:\n"
            for i, q in enumerate(conversation_history[:-1], 1):
                history_text += f"{i}. {q}\n"
            history_text += f"\nCurrent question: {question}"
        else:
            history_text = f"\n\nQuestion: {question}"
        
        # Simple follow-up prompt with conversation history
        followup_prompt = f"{QUESTION_PROMPT}\n\nYour previous analysis:\n{initial_answer}{history_text}"
        
        print(f"\n=== FOLLOW-UP QUESTION ===")
        print(f"Question: {question}")
        print(f"History length: {len(conversation_history) - 1}")
        print("="*50)
        
        # Generate answer to follow-up
        answer = self.generate(followup_prompt, 0.3)
        
        # Return both answer and updated history
        return {
            "answer": answer,
            "conversation_history": conversation_history
        }