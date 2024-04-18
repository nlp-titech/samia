
import numpy as np
from sklearn.metrics import roc_curve, auc
from utils import load_jsonl
import re
from collections import Counter
import zlib
import argparse

def get_suffix(text: str, prefix_ratio: float, text_length: int) -> list:
    """
    Extracts a suffix from the given text, based on the specified prefix ratio and text length.
    """
    words = text.split(" ")
    words = [word for word in words if word != ""]
    words = words[round(text_length*prefix_ratio):]
    return words

def ngrams(sequence, n) -> zip:
    """
    Generates n-grams from a sequence.
    """
    return zip(*[sequence[i:] for i in range(n)])

def rouge_n(candidate: list, reference: list, n=1) -> float:
    """
    Calculates the ROUGE-N score between a candidate and a reference.
    """
    if not candidate or not reference:
        return 0
    candidate_ngrams = list(ngrams(candidate, n))
    reference_ngrams = list(ngrams(reference, n))
    ref_words_count = Counter(reference_ngrams)
    cand_words_count = Counter(candidate_ngrams)
    overlap = ref_words_count & cand_words_count
    recall = sum(overlap.values()) / len(reference)
    precision = sum(overlap.values()) / len(candidate)
    return recall

def clean_text(text: str, model_name: str) -> str:
    """
    Removes specific special tokens from the text based on the model's output.
    """
    if model_name in {"gpt-j-6B", "pythia-6.9b"}:
        return re.sub(r'<\|endoftext\|>', '', text)
    elif model_name in {"Llama-2-7b", "opt-6.7b"}:
        text = re.sub(r'<s> ', '', text)
        return re.sub(r'</s>', '', text)
    return text

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="gpt-j-6B", choices=["gpt-j-6B", "opt-6.7b", "pythia-6.9b", "Llama-2-7b"], type=str)
    parser.add_argument("--text_length", default=32, choices=[32, 64, 128, 256], type=int)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--prefix_ratio", default=0.5, type=float)
    parser.add_argument("--zlib", action="store_true")
    parser.add_argument("--save", action="store_true")    
    args = parser.parse_args()
    
    model_name = args.model_name
    text_length = args.text_length
    num_samples = args.num_samples
    prefix_ratio = args.prefix_ratio

    lines_cand = load_jsonl(f"sample/{model_name}/{text_length}.jsonl")
    lines_ref = load_jsonl(f"wikimia/{text_length}.jsonl")

    rouge_seen, rouge_unseen = [], []
    for line_cand, line_ref in zip(lines_cand, lines_ref):
        suffix_ref = get_suffix(line_ref["input"], 0.5, text_length)
        rouge_scores = []
        for i in range(num_samples):
            text_output = clean_text(line_cand[f"output_{i}"], model_name)           
            suffix_cand = get_suffix(text_output, 0.5, text_length)
            if args.zlib:
                zlib_cand = zlib.compress(" ".join(suffix_cand).encode('utf-8'))
                rouge_scores.append(rouge_n(suffix_cand, suffix_ref, n=1) * len(zlib_cand))
            else:
                rouge_scores.append(rouge_n(suffix_cand, suffix_ref, n=1))
        (rouge_seen if line_ref["label"] else rouge_unseen).append(rouge_scores)
    
    # average over samples
    rouge_seen_avg = np.array(rouge_seen).mean(axis=1).tolist()
    rouge_unseen_avg = np.array(rouge_unseen).mean(axis=1).tolist()
    
    if args.save:
        if zlib:
            np.save(f"{args.model_name}_{args.text_length}_zlib_seen_avg",rouge_seen_avg)
            np.save(f"{args.model_name}_{args.text_length}_zlib_unseen_avg",rouge_unseen_avg)
        else:
            np.save(f"{args.model_name}_{args.text_length}_seen_avg",rouge_seen_avg)
            np.save(f"{args.model_name}_{args.text_length}_unseen_avg",rouge_unseen_avg)
    # calculate ROC-AUC
    y_true = [1] * len(rouge_seen_avg) + [0] * len(rouge_unseen_avg)
    y_score = rouge_seen_avg + rouge_unseen_avg
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    idx = np.argmin(np.abs(fpr - 0.10))

    print(f"ROC-AUC   : {roc_auc:.2f}")
    print(f"TPR@10%FPR: {tpr[idx]*100:.1f}%")
