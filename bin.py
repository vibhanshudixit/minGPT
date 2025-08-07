import os
import numpy as np
from tqdm import tqdm
import tiktoken

# Paths
train_path = "/mnt/d/TinyStories/TinyStories-train.txt"
valid_path = "/mnt/d/TinyStories/TinyStories-valid.txt"

# Tokenizer
enc = tiktoken.get_encoding("gpt2")
dtype = np.uint16

def tokenize_file(input_path, output_path):
    # First pass to count total tokens
    total_tokens = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_tokens += len(enc.encode_ordinary(line.strip()))

    # Allocate memmap
    arr = np.memmap(output_path, dtype=dtype, mode='w+', shape=(total_tokens,))
    
    # Second pass to tokenize and write
    idx = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Writing {os.path.basename(output_path)}"):
            ids = enc.encode_ordinary(line.strip())
            ids = np.array(ids, dtype=dtype)
            arr[idx:idx + len(ids)] = ids
            idx += len(ids)
    arr.flush()

# Convert both files
tokenize_file(train_path, "train.bin")
tokenize_file(valid_path, "valid.bin")
