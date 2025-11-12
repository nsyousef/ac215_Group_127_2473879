import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM
from IPython.display import display, Markdown
from transformers import BitsAndBytesConfig 
class LLM:
    """
    Minimal local wrapper for the MedGemma-4B instruction-tuned model.
    Loads model + processor once, maintains a base prompt,
    and builds contextual prompts with predictions/confidences.
    """

    def __init__(self,
                 model_name: str = "medgemma-4b",
                 device: str = None,
                 dtype: torch.dtype = torch.bfloat16,
                 max_new_tokens: int = 1024,
                 base_prompt: str = None):
        # Device detection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        if model_name == "medgemma-4b":
            print(f"Loading model ({model_name}) on {self.device}...")
            model_id = "google/medgemma-4b-it"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=self.dtype,
                device_map="auto",
            )
            self.model.to(self.device)

        elif model_name == 'medgemma-27b':
            print(f"Loading model ({model_name}) on {self.device}...")
            model_id = "google/medgemma-27b-text-it"
            self.processor = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=self.dtype,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True)
            )
            self.model.to(self.device)
        # Format base prompt: strip leading/trailing whitespace but preserve structure
        if base_prompt is not None:
            # Strip leading/trailing whitespace and normalize multiple spaces/newlines to single space
            # But keep it as a single line to avoid tokenization issues
            self.base_prompt = ' '.join(base_prompt.split())
        else:
            self.base_prompt = ""

    def build_prompt(self, predictions: dict, metadata: dict) -> str:
        """
        Combine the base prompt with model predictions and metadata.
        predictions: dictionary of predicted conditions and their confidence scores
        history: dictionary of history of conditions and their confidence scores
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
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)

        token_count = inputs['input_ids'].shape[1]
        
        outputs = self.model.generate(
            **inputs,  # Unpack the inputs dict
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None
        )
        input_length = inputs['input_ids'].shape[1]

        # Decode only the NEW tokens (skip the input)
        generated_ids = outputs[0][input_length:]  # Slice off the input tokens
        output_length = len(generated_ids) - input_length
        print(f"Input length: {input_length}, Output length: {output_length}")
        text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        self.display_markdown(text)
        return text

    def explain(self, predictions: dict, metadata: dict, temperature: float = 0.3) -> str:
        """
        Full convenience method: build the prompt + run the model.
        predictions: dictionary of predicted conditions and their confidence scores
        metadata: dictionary of metadata
        """
        prompt = self.build_prompt(predictions, metadata)
        answer = self.generate(prompt, temperature)
        return answer

    def display_markdown(self, text: str):
        display(Markdown(text))