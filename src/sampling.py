from model_loader import load_model
from tqdm import tqdm
from utils import load_jsonl, add_jsonl
import argparse
from huggingface_hub import login

YOUR_HUGGINGFACE_TOKEN=""
login(YOUR_HUGGINGFACE_TOKEN)

def generate_text(model, tokenizer, prompt: str, gpu) -> str:
    tokenizer.pad_token = tokenizer.eos_token
    encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = encoding.input_ids.to(gpu)
    attention_mask = encoding.attention_mask.to(gpu)

    gen_token = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=1,
        max_length=1024, # input+output
        top_k=50,
        top_p=1,
    )
    gen_text = tokenizer.batch_decode(gen_token)[0]
    return gen_text

def get_prefix(text: str, prefix_ratio: float) -> str:
    num_words = len(text.split())
    num_prefix_words = int(num_words * prefix_ratio)
    prefix = " ".join(text.split()[:num_prefix_words])
    return prefix

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="gpt-j-6B", choices=["gpt-j-6B", "opt-6.7b", "pythia-6.9b", "Llama-2-7b"], type=str)
    parser.add_argument("--text_length", default=32, choices=[32, 64, 128, 256], type=int)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--prefix_ratio", default=0.5, type=float)
    
    args = parser.parse_args()
    
    model_name = args.model_name
    text_length = args.text_length
    num_samples = args.num_samples
    prefix_ratio = args.prefix_ratio

    model, tokenizer = load_model(model_name, "cuda:0")

    lines = load_jsonl(f"./wikimia/{text_length}.jsonl")
    for line in tqdm(lines):
        new_line = {}
        prefix = get_prefix(line["input"], prefix_ratio=prefix_ratio)
        new_line["input"] = prefix
        for i in range(num_samples):
            new_line[f"output_{i}"] = generate_text(model, tokenizer, prefix, "cuda:0")

        add_jsonl(new_line, f"./sample/{model_name}/{text_length}.jsonl")

