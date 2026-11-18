import os
import sys
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# settings
DATA_FILE = "input.txt"  # plain text file to learn from
CKPT_FILE = "mini_llm.pt"  # where the model is saved
BLOCK_SIZE = 256  # how many characters the model can look back at
BATCH_SIZE = 32
N_EMBD = 256  # width of the model
N_HEAD = 8  # attention heads
N_LAYER = 6  # transformer blocks
DROPOUT = 0.1
LR = 3e-4
MAX_ITERS = 5000
EVAL_EVERY = 500
EVAL_ITERS = 50
SAVE_EVERY = 500

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


# model
class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.resid_drop = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.register_buffer("mask", mask.view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        head_size = C // N_HEAD
        q = q.view(B, T, N_HEAD, head_size).transpose(1, 2)  # (B, heads, T, head_size)
        k = k.view(B, T, N_HEAD, head_size).transpose(1, 2)
        v = v.view(B, T, N_HEAD, head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)  # how much each token looks at others
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # no peeking at the future
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # weighted mix of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):


    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual connections
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)  # character -> vector
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)  # position -> vector
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)  # vector -> score for each character

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# data
def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], chars


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])  # targets are inputs shifted by one
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    results = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses[k] = loss.item()
        results[name] = losses.mean().item()
    model.train()
    return results


# train
def train():
    if not os.path.exists(DATA_FILE):
        sys.exit(f"Can't find {DATA_FILE}. Download a text file and name it {DATA_FILE}.")

    train_data, val_data, chars = load_data()
    model = MiniGPT(len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"device: {device}")
    print(f"vocab size: {len(chars)} characters")
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    start_it = 1
    if os.path.exists(CKPT_FILE):  # resume from the last checkpoint
        ck = torch.load(CKPT_FILE, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_it = ck["step"] + 1
        print(f"resumed from checkpoint at step {ck['step']}")

    t0 = time.time()
    for it in range(start_it, MAX_ITERS + 1):
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % EVAL_EVERY == 0 or it == start_it:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {it:5d} | train loss {losses['train']:.3f} | "
                  f"val loss {losses['val']:.3f} | {time.time() - t0:.0f}s")

        if it % SAVE_EVERY == 0 or it == MAX_ITERS:
            torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(),
                        "step": it, "chars": chars}, CKPT_FILE)

    print(f"done. saved to {CKPT_FILE}")


# generate
def generate_text():
    if not os.path.exists(CKPT_FILE):
        sys.exit(f"No {CKPT_FILE} found. Run 'python mini_llm.py train' first.")

    ck = torch.load(CKPT_FILE, map_location=device)
    chars = ck["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    model = MiniGPT(len(chars)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    print("Type a starting prompt and press Enter. Type 'quit' to exit.\n")
    while True:
        prompt = input("prompt> ")
        if prompt.strip().lower() == "quit":
            break
        ctx = [stoi[c] for c in prompt if c in stoi] or [0]
        idx = torch.tensor([ctx], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=400)[0].tolist()
        print("\n" + "".join(itos[i] for i in out) + "\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "train":
        train()
    elif mode == "generate":
        generate_text()
    else:
        print("Usage: python mini_llm.py [train|generate]")
