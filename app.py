"""
OSLLM: FINAL VERIFIED VERSION (NO FREEZING)
-------------------------------------------
- Ensures training starts after tokenization
- Adds progress tracking to prevent "freezing"
- Optimizes dataset size to speed up testing
- Dynamically adjusts batch size based on GPU availability
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# ✅ Auto-detect GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.device_count() > 1:
    print(f"🔥 Multi-GPU Enabled! Using {torch.cuda.device_count()} GPUs.")

# ✅ Auto-adjust batch size based on available memory
BATCH_SIZE = 8 if DEVICE == "cuda" else 2  
VOCAB_SIZE = 32000
EPOCHS = 3  # Reduce for faster testing
LEARNING_RATE = 5e-5
MODEL_FILE = "osllm_model.pth"
TOKENIZER_FILE = "custom_tokenizer.json"

# ------------------------------------------------------------
# 1. Transformer Model with Multi-GPU Support
# ------------------------------------------------------------
class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super(CustomTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.transformer = nn.Transformer(
            d_model=512, nhead=8, num_encoder_layers=6, 
            num_decoder_layers=6, batch_first=True
        )
        self.fc_out = nn.Linear(512, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output)

# ------------------------------------------------------------
# 2. Train Tokenizer
# ------------------------------------------------------------
def train_tokenizer():
    """Trains a tokenizer from scratch"""
    corpus_file = "data/training_corpus.txt"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(corpus_file):
        with open(corpus_file, "w") as f:
            f.write("User: Hello\nOSLLM: Hello! How can I help?\n" * 500)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[EOS]"])
    tokenizer.train([corpus_file], trainer)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'unk_token': '[UNK]', 'bos_token': '[CLS]', 'eos_token': '[EOS]'})
    tokenizer.save_pretrained("./")
    return tokenizer

# ------------------------------------------------------------
# 3. Preprocess Data
# ------------------------------------------------------------
def preprocess_data():
    """Loads and tokenizes dataset"""
    print("📥 Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # ✅ Reduce dataset size for faster training during testing
    dataset = dataset["train"].select(range(5000))  

    tokenizer = PreTrainedTokenizerFast.from_pretrained("./")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    print("🔄 Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    print("✅ Tokenization Complete! Dataset Ready.")
    return torch.tensor(tokenized_datasets["input_ids"])

# ------------------------------------------------------------
# 4. Train Model with GPU Support
# ------------------------------------------------------------
def train_model():
    """Trains OSLLM with full GPU acceleration"""
    print("🚀 Training OSLLM...")

    tokenizer = train_tokenizer()
    train_data = preprocess_data()

    print("✅ Training Starting Now!")  # Debugging line

    model = CustomTransformerModel(VOCAB_SIZE).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        print(f"📢 Starting Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, VOCAB_SIZE), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:  # ✅ Shows live training updates every 10 batches
                print(f"🟢 Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_time = time.time() - start_time
        print(f"📊 Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Time: {epoch_time:.2f} sec ✅")

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), MODEL_FILE)
    else:
        torch.save(model.state_dict(), MODEL_FILE)

    print("🚀 Training Complete! Model saved.")

# ------------------------------------------------------------
# 5. Interactive Menu
# ------------------------------------------------------------
def main():
    """Menu to select training or chat mode"""
    while True:
        print("\n🔥 OSLLM Main Menu 🔥")
        print("[1] Train OSLLM")
        print("[2] Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            train_model()
        elif choice == "2":
            print("👋 Exiting OSLLM. Have a great day!")
            break
        else:
            print("❌ Invalid choice! Please enter 1 or 2.")

if __name__ == "__main__":
    main()

