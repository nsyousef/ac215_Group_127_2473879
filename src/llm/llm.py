import re
import time
import threading
import torch
from typing import Generator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


class ThoughtStoppingCriteria(StoppingCriteria):
    """Stop generation when 'thought' token is encountered."""

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.thought_tokens = tokenizer.encode("thought", add_special_tokens=False)
        self.thought_token = self.thought_tokens[0] if self.thought_tokens else None

    def __call__(self, input_ids, scores, **kwargs):
        if self.thought_token is not None:
            generated_ids = input_ids[0][self.prompt_length :]
            if len(generated_ids) > 0 and generated_ids[-1] == self.thought_token:
                return True
        return False


class LLM:
    """LLM wrapper for dermatology assistant with Modal deployment support."""

    def __init__(
        self,
        model_name: str = "medgemma-27b",
        max_new_tokens: int = 700,
        base_prompt: str = "",
        question_prompt: str = "",
        time_tracking_prompt: str = "",
    ):
        self.max_new_tokens = max_new_tokens
        self.base_prompt = " ".join(base_prompt.split()) if base_prompt else ""
        self.question_prompt = " ".join(question_prompt.split()) if question_prompt else ""
        self.time_tracking_prompt = " ".join(time_tracking_prompt.split()) if time_tracking_prompt else ""
        self.model_name = model_name

        print(f"Loading model: {model_name}")

        if model_name == "medgemma-4b":
            model_id = "google/medgemma-4b-it"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        elif model_name == "medgemma-27b":
            model_id = "google/medgemma-27b-text-it"
            self.processor = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")

    def _get_tokenizer_for_streamer(self):
        """Get the tokenizer for TextIteratorStreamer."""
        # For medgemma-27b, processor is already a tokenizer
        # For medgemma-4b, processor has a tokenizer attribute
        if hasattr(self.processor, "tokenizer"):
            return self.processor.tokenizer
        return self.processor

    def build_prompt(self, predictions: dict, metadata: dict) -> str:
        """Build prompt from predictions and metadata."""
        summary = [f"{pred} ({conf*100:.1f}%)" for pred, conf in predictions.items()]
        prompt = f"{self.base_prompt}\n\nINPUT DATA: {', '.join(summary)}.\n"

        if "user_input" in metadata:
            prompt += f"User input: {metadata['user_input']}\n"

        # New: include high-level CV time-tracking summary text instead of raw metrics
        cv_tracking_summary = metadata.get("cv_tracking_summary")
        if cv_tracking_summary:
            prompt += f"\nTime tracking CV summary:\n{cv_tracking_summary}\n"

        if "history" in metadata and metadata["history"]:
            prompt += "\nHistorical Data:\n"
            for date, data in sorted(metadata["history"].items()):
                prompt += f"Date: {date}.\n"
                if "text_summary" in data:
                    prompt += f"  User note: {data['text_summary']}.\n"
                if "tracking_summary" in data and data["tracking_summary"]:
                    prompt += f"  Tracking summary: {data['tracking_summary']}.\n"

        return prompt

    @torch.inference_mode()
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate response with thought stopping."""
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(formatted_prompt, return_tensors="pt", add_special_tokens=True).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        stopping_criteria = StoppingCriteriaList([ThoughtStoppingCriteria(self.processor, input_length)])

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.processor.pad_token_id,
            eos_token_id=self.processor.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

        generated_ids = outputs[0][input_length:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        return self._extract_user_response(text)

    def _extract_user_response(self, text: str) -> str:
        """Remove thought markers and their content."""
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)

        if "thought" in text.lower():
            thought_match = re.search(r"\bthought\b", text, re.IGNORECASE)
            if thought_match:
                text = text[: thought_match.start()]

        if "<thought>" in text.lower():
            thought_pos = text.lower().find("<thought>")
            text = text[:thought_pos]

        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def explain(self, predictions: dict, metadata: dict, temperature: float = 0.3) -> str:
        """Generate explanation from predictions and metadata."""
        prompt = self.build_prompt(predictions, metadata)
        return self.generate(prompt, temperature)

    def ask_followup(
        self, initial_answer: str, question: str, conversation_history: list = None, temperature: float = 0.3
    ) -> dict:
        """Answer follow-up question with conversation history."""
        if conversation_history is None:
            conversation_history = []

        conversation_history = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
        conversation_history.append(question)

        history_text = ""
        if len(conversation_history) > 1:
            history_text = "\n\nPrevious questions in this conversation:\n"
            for i, q in enumerate(conversation_history[:-1], 1):
                history_text += f"{i}. {q}\n"
            history_text += f"\nCurrent question: {question}"
        else:
            history_text = f"\n\nQuestion: {question}"

        followup_prompt = f"{self.question_prompt}\n\nYour previous analysis:\n{initial_answer}{history_text}"
        answer = self.generate(followup_prompt, temperature)

        return {"answer": answer, "conversation_history": conversation_history}

    @torch.inference_mode()
    def generate_stream(self, prompt: str, temperature: float = 0.3) -> Generator[str, None, None]:
        """
        Stream response chunks while preserving the 'thought' filtering logic.
        Yields plain text chunks suitable for streaming over HTTP.
        """
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]
        stopping_criteria = StoppingCriteriaList([ThoughtStoppingCriteria(self.processor, input_length)])

        streamer = TextIteratorStreamer(
            self._get_tokenizer_for_streamer(),
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.processor.pad_token_id,
            eos_token_id=self.processor.eos_token_id,
            stopping_criteria=stopping_criteria,
            streamer=streamer,
        )

        # Run model.generate in a background thread so we can iterate over streamer
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # We use the same cleanup logic as in the non-streaming path,
        # but incrementally: keep an accumulated buffer and only send
        # the new cleaned suffix each time.
        accumulated = ""
        already_sent = 0

        for new_text in streamer:
            if not new_text:
                continue

            accumulated += new_text
            cleaned = self._extract_user_response(accumulated)

            # Only send what hasn't been sent yet
            to_send = cleaned[already_sent:]
            if to_send:
                already_sent = len(cleaned)
                yield to_send

    def ask_followup_stream(
        self,
        initial_answer: str,
        question: str,
        conversation_history: list = None,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """Streaming variant of ask_followup()."""
        if conversation_history is None:
            conversation_history = []

        conversation_history = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
        conversation_history.append(question)

        history_text = ""
        if len(conversation_history) > 1:
            history_text = "\n\nPrevious questions in this conversation:\n"
            for i, q in enumerate(conversation_history[:-1], 1):
                history_text += f"{i}. {q}\n"
            history_text += f"\nCurrent question: {question}"
        else:
            history_text = f"\n\nQuestion: {question}"

        followup_prompt = f"{self.question_prompt}\n\nYour previous analysis:\n{initial_answer}{history_text}"
        return self.generate_stream(followup_prompt, temperature)

    def explain_stream(self, prompt: str, on_chunk):
        """
        Stream the explanation for a prediction.

        Args:
            prompt: The constructed explanation prompt (string)
            on_chunk: A callback that receives each incremental text chunk (string)

        Behavior:
            - Calls self.generate_stream(prompt) to get incremental tokens
            - Sends each token to on_chunk()
            - Does NOT return the full text (caller assembles final answer)
        """
        if on_chunk is None:
            raise ValueError("explain_stream requires on_chunk callback")

        try:
            # generate_stream() yields token pieces from your LLM backend
            for delta in self.generate_stream(prompt):
                if delta:
                    on_chunk(delta)

        except Exception as e:
            raise RuntimeError(f"Streaming explain failed: {e}")

    def explain_stream_generator(self, prompt: str):
        """
        Generator used by the Modal/HTTP streaming endpoint.
        Yields dicts of the form: {"delta": "..."}.
        """
        for delta in self.generate_stream(prompt):
            if delta:
                yield {"delta": delta}

    def time_tracking_summary(self, user_input: str, cv_analysis_history: dict, temperature: float = 0.3) -> dict:
        """
        Generate a brief summary of time tracking data changes.

        Args:
            user_input: User's text description of the condition
            cv_analysis_history: Date-keyed CV metrics {"2024-12-01": {...}, ...}
            temperature: Generation temperature

        Returns:
            {"summary": "3-4 sentence summary text"}
        """
        # Build prompt for time tracking
        prompt = f"{self.time_tracking_prompt}\n\n"

        if user_input:
            prompt += f"User description: {user_input}\n\n"

        prompt += "Tracking Data:\n"
        sorted_dates = sorted(cv_analysis_history.keys())

        for i, date in enumerate(sorted_dates):
            metrics = cv_analysis_history[date]
            prompt += f"\n{'First entry' if i == 0 else f'Entry {i+1}'} ({date}):\n"

            if "area_cm2" in metrics and metrics["area_cm2"] is not None:
                prompt += f"  - Area: {metrics['area_cm2']:.2f} cm²\n"

            if "compactness_index" in metrics:
                prompt += f"  - Shape compactness: {metrics['compactness_index']:.2f}\n"

            if "color_stats_lab" in metrics:
                color = metrics["color_stats_lab"]
                prompt += f"  - Color (LAB): L={color.get('mean_L', 0):.1f}, A={color.get('mean_A', 0):.1f}, B={color.get('mean_B', 0):.1f}\n"

        # Generate summary
        summary_text = self.generate(prompt, temperature)

        return {"summary": summary_text}
