"""
MCP Corpus Preparation - Converts crawled MCP data into training format.
Replaces autoresearch's default prepare.py data pipeline for MCP-specific training.

Reads text files from corpus/, trains a BPE tokenizer, and creates
train/val parquet shards in the same format autoresearch expects.

Usage: python3 prepare_mcp.py
"""

import os
import sys
import math
import time
import random
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (tuned for smaller MCP corpus)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024       # shorter context for MCP content (vs 2048 for general text)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_TOKENS = 10 * 524288 # reduced eval tokens for smaller dataset
VOCAB_SIZE = 4096         # smaller vocab for domain-specific data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORPUS_DIR = "corpus"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-mcp")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# BPE split pattern (GPT-4 style)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

VAL_FILENAME = "shard_val.parquet"

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus_texts():
    """Load all text files from the corpus directory."""
    texts = []
    corpus_dir = CORPUS_DIR
    if not os.path.isdir(corpus_dir):
        print(f"Error: corpus directory '{corpus_dir}' not found. Run mcp_researcher.py first.")
        sys.exit(1)

    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if content.strip():
            texts.append(content)
            print(f"  Loaded {filename}: {len(content):,} chars")

    if not texts:
        print("Error: no text files found in corpus/. Run mcp_researcher.py first.")
        sys.exit(1)

    return texts


def split_into_documents(texts, max_doc_len=2000):
    """Split corpus texts into individual documents for training."""
    documents = []
    for text in texts:
        # Split on double newlines (paragraph boundaries)
        chunks = text.split("\n\n")
        current_doc = ""
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if len(current_doc) + len(chunk) + 2 > max_doc_len and current_doc:
                documents.append(current_doc)
                current_doc = chunk
            else:
                current_doc = current_doc + "\n\n" + chunk if current_doc else chunk
        if current_doc:
            documents.append(current_doc)

    random.shuffle(documents)
    return documents


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def train_tokenizer(documents):
    """Train BPE tokenizer on MCP corpus."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print("Tokenizer: training BPE tokenizer on MCP corpus...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)

    # Create an iterator over documents
    def doc_iterator():
        for doc in documents:
            yield doc

    tokenizer.train_from_iterator(doc_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe_mcp",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, vocab_size={enc.n_vocab}, saved to {tokenizer_pkl}")

    # Build token_bytes lookup for BPB evaluation
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "MCP server for database access with 500 stars on GitHub"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


# ---------------------------------------------------------------------------
# Shard creation
# ---------------------------------------------------------------------------

def create_shards(documents, val_ratio=0.1, docs_per_shard=500):
    """Tokenize documents and create parquet shards."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load tokenizer
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    with open(tokenizer_pkl, "rb") as f:
        enc = pickle.load(f)

    bos_id = enc.encode_single_token(BOS_TOKEN)

    # Split train/val
    n_val = max(1, int(len(documents) * val_ratio))
    val_docs = documents[:n_val]
    train_docs = documents[n_val:]

    print(f"Shards: {len(train_docs)} train docs, {len(val_docs)} val docs")

    def write_shard(docs, filepath):
        """Tokenize docs and write as parquet shard."""
        texts = []
        for doc in docs:
            # Prepend BOS to each doc text (tokenizer will handle encoding)
            texts.append(doc)
        table = pa.table({"text": texts})
        pq.write_table(table, filepath)

    # Write val shard
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    write_shard(val_docs, val_path)
    print(f"  Wrote val shard: {val_path} ({len(val_docs)} docs)")

    # Write train shards
    shard_idx = 0
    for i in range(0, len(train_docs), docs_per_shard):
        batch = train_docs[i:i + docs_per_shard]
        shard_path = os.path.join(DATA_DIR, f"shard_{shard_idx:05d}.parquet")
        write_shard(batch, shard_path)
        print(f"  Wrote train shard {shard_idx}: {shard_path} ({len(batch)} docs)")
        shard_idx += 1

    print(f"Shards: created {shard_idx} train shards + 1 val shard in {DATA_DIR}")
    return shard_idx


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py when using MCP data)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper compatible with autoresearch's train.py."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare_mcp.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=200):
    """BOS-aligned dataloader with best-fit packing (same interface as prepare.py)."""
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """Bits per byte evaluation (same interface as prepare.py)."""
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"MCP Corpus Preparation")
    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Load corpus
    print("Step 1: Loading corpus...")
    texts = load_corpus_texts()
    total_chars = sum(len(t) for t in texts)
    print(f"  Total: {len(texts)} files, {total_chars:,} characters")
    print()

    # Step 2: Split into documents
    print("Step 2: Splitting into documents...")
    documents = split_into_documents(texts)
    print(f"  Created {len(documents)} documents")
    print()

    # Step 3: Train tokenizer
    print("Step 3: Training tokenizer...")
    train_tokenizer(documents)
    print()

    # Step 4: Create shards
    print("Step 4: Creating parquet shards...")
    num_shards = create_shards(documents)
    print()

    print("Done! MCP training data ready.")
    print(f"  Tokenizer: {TOKENIZER_DIR}")
    print(f"  Data shards: {DATA_DIR}")
    print(f"  MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    print(f"  VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"  EVAL_TOKENS: {EVAL_TOKENS}")
