import torch
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

class ModelTier(Enum):
    FASTEST = "fastest"
    BALANCED = "balanced"
    QUALITY = "quality"
    BEST = "best"

MODEL_CONFIGS = {
    ModelTier.FASTEST: {
        "name": "microsoft/Phi-3-mini-128k-instruct",
        "vram_usage": "~2.5GB",
        "expected_speed": "50+ tok/s",
    },
    ModelTier.BALANCED: {
        "name": "microsoft/Phi-3-medium-4k-instruct",
        "vram_usage": "~5.5GB",
        "expected_speed": "35+ tok/s",
    },
    ModelTier.QUALITY: {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "vram_usage": "~5.7GB",
        "expected_speed": "20-25 tok/s",
    },
    ModelTier.BEST: {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "vram_usage": "~5.8GB",
        "expected_speed": "25+ tok/s",
    }
}

class LLMRunner:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_tier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def check_system(self):
        print("System check...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {torch.cuda.get_device_name(0)} with {total_vram:.1f} GB VRAM")
            if total_vram < 6:
                print("Warning: Less than 6GB VRAM, some models may not fit.")
            return total_vram >= 4.0
        else:
            print("CUDA not available! Requires GPU.")
            return False

    def clear_memory(self):
        if self.model:
            del self.model, self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            self.model = None
            self.tokenizer = None

    def load_model(self, tier: ModelTier):
        if not self.check_system():
            return False
        self.clear_memory()

        config = MODEL_CONFIGS[tier]
        model_name = config["name"]
        print(f"\nLoading {tier.value} model: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            print("Loading model with 4-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            self.model.eval()
            self.current_tier = tier

            mem_alloc = torch.cuda.memory_allocated(0) / 1e9
            mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"Model loaded! VRAM usage: {mem_alloc:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def generate_text(self, prompt, max_tokens=200):
        if not self.model:
            print("No model loaded!")
            return ""

        model_name = MODEL_CONFIGS[self.current_tier]["name"].lower()
        if "phi-3" in model_name:
            formatted_prompt = prompt
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

        print(f"\nPrompt: {prompt}")
        print("=" * 40)
        print("Response:")

        start = time.time()

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode full output tokens
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Decode prompt tokens exactly for slicing
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)

        # Remove prompt part from start of full_text cleanly
        if full_text.startswith(prompt_text):
            response = full_text[len(prompt_text):].strip()
        else:
            # fallback if tokenizer handles special tokens differently
            response = full_text.strip()

        print(response)

        duration = time.time() - start
        speed = len(response.split()) / duration if duration > 0 else 0
        print(f"\nPerformance: {speed:.1f} tokens/sec | {duration:.2f}s | Model: {self.current_tier.value}")

        return response

    def interactive_mode(self):
        if not self.model:
            print("Load a model first!")
            return

        print(f"\nInteractive mode: {self.current_tier.value}")
        print("Commands: 'quit', 'switch', 'memory'")
        while True:
            try:
                inp = input("\nYou: ").strip()
                if inp.lower() == "quit":
                    print("Bye!")
                    break
                elif inp.lower() == "switch":
                    print("Choose model:")
                    for i, tier in enumerate(ModelTier, 1):
                        print(f"{i}. {tier.value}")
                    choice = input("Select: ").strip()
                    tier_map = {str(i): tier for i, tier in enumerate(ModelTier, 1)}
                    tier = tier_map.get(choice)
                    if tier and self.load_model(tier):
                        print(f"Switched to {tier.value}")
                    else:
                        print("Invalid choice or failed to load model")
                elif inp.lower() == "memory":
                    mem_alloc = torch.cuda.memory_allocated(0) / 1e9
                    mem_cached = torch.cuda.memory_reserved(0) / 1e9
                    print(f"VRAM: {mem_alloc:.2f} GB allocated, {mem_cached:.2f} GB cached")
                elif inp == "":
                    continue
                else:
                    self.generate_text(inp, max_tokens=250)
            except KeyboardInterrupt:
                print("\nInterrupted. Bye!")
                break
            except torch.cuda.OutOfMemoryError:
                print("Out of GPU memory! Try switching to smaller model.")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error: {e}")

def main():
    runner = LLMRunner()
    if not runner.check_system():
        return

    print("\nChoose model tier:")
    for i, tier in enumerate(ModelTier, 1):
        print(f"{i}. {tier.value.capitalize()}")

    choice = input("Enter choice (1-4): ").strip()
    tier_map = {str(i): tier for i, tier in enumerate(ModelTier, 1)}
    tier = tier_map.get(choice, ModelTier.BALANCED)
    if runner.load_model(tier):
        runner.interactive_mode()
    else:
        print("Failed to load model")

if __name__ == "__main__":
    main()
