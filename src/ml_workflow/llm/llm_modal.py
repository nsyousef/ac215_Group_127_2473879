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

from prompts import BASE_PROMPT, EVIL_PROMPT
secrets = [modal.Secret.from_name("hf")]
HF_CACHE = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App(name="testing-llm-modal", 
                image=llm_image, 
                secrets=secrets,
                volumes={
                    "/root/.cache/huggingface": HF_CACHE,
                }
                )

@app.cls(gpu = 'A100-80GB', min_containers = 0, max_containers = 1, scaledown_window = 10)
class LLM:
    @modal.enter()
    def initialize_model(self):
        # Device detection
        self.dtype = torch.bfloat16
        self.max_new_tokens = 2048
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
            self.model.to(device)

        elif model_name == 'medgemma-27b':
            print(f"Loading model ({model_name})")
            model_id = "google/medgemma-27b-text-it"
            self.processor = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=self.dtype,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True)
            )
            self.model.to(device)
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


        prompt = f"{self.base_prompt}\n Model results for predicted Conditions: {summary_str}.\n"

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
            else:
                prompt += f"Latest CV Analysis: {cv}.\n"

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
                    prompt += f"CV Analysis: {', '.join(parts)}.\n"
            if 'text_summary' in data:
                prompt += f"Text Summary: {data['text_summary']}.\n"
        return prompt
        
    @torch.inference_mode()
    def get_input_ids(self, prompt: str) -> torch.Tensor:
        """Get input IDs for the model."""
        return self.processor(text=prompt, return_tensors="pt").to(self.model.device)

    @torch.inference_mode()
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate text output from the model given a prepared prompt."""
        # For MedGemma, use simple text input without chat template
        start_time = time.time()
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        end_time = time.time()
        print(f"Time taken to get input IDs: {end_time - start_time} seconds")
        token_count = inputs['input_ids'].shape[1]
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,  # Unpack the inputs dict
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None
        )
        end_time = time.time()
        print(f"Time taken to generate: {end_time - start_time} seconds")
        input_length = inputs['input_ids'].shape[1]

        # Decode only the NEW tokens (skip the input)
        start_time = time.time()
        generated_ids = outputs[0][input_length:]  # Slice off the input tokens
        output_length = len(generated_ids) - input_length
        print(f"Input length: {input_length}, Output length: {output_length}")
        text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        end_time = time.time()
        print(f"Time taken to decode: {end_time - start_time} seconds")
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