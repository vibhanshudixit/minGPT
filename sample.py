"""
Sample from a trained GPT-2 model on RTX 4060 (TinyStories)
"""
import os
import pickle
import torch
import tiktoken
from contextlib import nullcontext
from minGPT import GPT, GPTConfig  # use model from nanoGPT

# ---- Inference Settings ----
out_dir = 'out'
start = "Once upon a time"  # prompt to prime the model
num_samples = 3
max_new_tokens = 100
temperature = 0.8
top_k = 100
seed = 42

# ---- Device Setup ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# ---- Load Checkpoint ----
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(checkpoint['model'])
model.eval().to(device)

# ---- Tokenizer Setup ----
meta_path = os.path.join('data', checkpoint.get('config', {}).get('dataset', ''), 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# ---- Prompt Encoding ----
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]

# ---- Generate and Print Samples ----
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

with torch.no_grad():
    with ctx:
        for i in range(num_samples):
            y = model.generate(start_ids, max_new_tokens, temperature=temperature, top_k=top_k)
            print(f"\n--- Sample {i+1} ---")
            print(decode(y[0].tolist()))
