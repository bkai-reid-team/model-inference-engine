from ray import serve
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@serve.deployment(name="qwen_api")
class QwenAPI:
    def __init__(self):
        print("🔹 Loading Qwen model...")
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    async def generate(self, body: dict):
        prompt = body.get("prompt", "")
        if not prompt:
            return {"error": "Missing 'prompt'"}
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=body.get("max_tokens", 256))
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": text}


def app(config: dict | None = None):
    # Only bind internal deployment (no HTTP routes here)
    return QwenAPI.bind()
