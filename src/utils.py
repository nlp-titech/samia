import json

def load_jsonl(path):
    with open(path, "r") as f:
        lines = [json.loads(l) for l in f.readlines()]
        return lines

def add_jsonl(new_line, path):
    with open(path, "a") as f:
        json.dump(new_line, f)
        f.write("\n")
