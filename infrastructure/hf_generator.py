from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from domain import IGenerator

class HuggingFaceGenerator(IGenerator):
    def __init__(self, model_id: str = "deepseek-ai/deepseek-llm-7b-chat"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config
        )
    def generate(self, formatted_prompt: str) -> str:
        messages = [{"role": "system", "content": "You are GymRat, a helpful assistant specialized in fitness, "
                                                  "workout routines, muscle growth, fat loss, and nutrition.\n\n"
                                                  "Below are previously answered questions and their expert responses. "
                                                  "Use them as context to answer the new question accurately.\n\n"},
                    {"role": "user", "content": formatted_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                prompt_text,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = outputs[0][prompt_text.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
