"""
OSLLM: Blank-Slate Large Language Model (LLM)
------------------------------------------------
This script trains a custom LLM from scratch using a transformer-based model.
- No pre-trained weights are used.
- Uses a Byte-Pair Encoding (BPE) tokenizer.
- Implements a simple transformer-based architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# Global Configuration
VOCAB_SIZE = 32000
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# 1. Custom Transformer Model Definition
# ------------------------------------------------------------
class CustomTransformerModel(nn.Module):
    """A custom transformer-based LLM with random initialization."""
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, d_model))  # Learnable positional encodings

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout
        )

        self.fc_out = nn.Linear(d_model, vocab_size)  # Output layer
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, tgt):
        """Forward pass through the transformer model."""
        src_embedded = self.embedding(src) + self.positional_encoding[:, :src.shape[1], :]
        tgt_embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.shape[1], :]

        output = self.transformer(src_embedded.permute(1, 0, 2), tgt_embedded.permute(1, 0, 2))
        output = self.fc_out(output.permute(1, 0, 2))
        return self.softmax(output)


# ------------------------------------------------------------
# 2. Tokenizer Training from Scratch (Byte-Pair Encoding)
# ------------------------------------------------------------
def train_custom_tokenizer(corpus_file="data/training_corpus.txt"):
    """Train a Byte-Pair Encoding (BPE) tokenizer from scratch."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=2, show_progress=True,
        special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"]
    )

    # Train tokenizer on corpus
    tokenizer.train([corpus_file], trainer)
    tokenizer.save("custom_tokenizer.json")

    print("✅ Custom Tokenizer Trained & Saved.")
    return PreTrainedTokenizerFast(tokenizer_file="custom_tokenizer.json")


# ------------------------------------------------------------
# 3. Data Preprocessing (Tokenization & Conversion)
# ------------------------------------------------------------
def preprocess_data():
    """Load dataset, tokenize it, and convert it into PyTorch tensors."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Load trained tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom_tokenizer.json")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Convert to tensor format
    train_data = torch.tensor(tokenized_datasets["train"]["input_ids"])
    print("✅ Training Data Processed.")
    return train_data


# ------------------------------------------------------------
# 4. Training Loop
# ------------------------------------------------------------
def train_model(model, train_data):
    """Train the transformer model from scratch."""
    train_dataset = TensorDataset(train_data[:-1], train_data[1:])  # Shift for next-word prediction
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            src, tgt = batch[0].to(DEVICE), batch[1].to(DEVICE)

            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, VOCAB_SIZE), tgt.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📊 Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} ✅")

    print("🚀 Training Completed!")


# ------------------------------------------------------------
# 5. Inference / Text Generation
# ------------------------------------------------------------
def generate_text(model, tokenizer, prompt, max_length=100):
    """Generate text using the trained model."""
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    output = model(tokens, tokens)  # Using itself as target
    output_text = tokenizer.decode(torch.argmax(output, dim=-1)[0])
    return output_text


# ------------------------------------------------------------
# 6. Main Execution (Building the OSLLM Model)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Initializing OSLLM Training...")

    # Train the tokenizer if not already trained
    tokenizer = train_custom_tokenizer()

    # Load & process training data
    train_data = preprocess_data()

    # Initialize blank transformer model
    model = CustomTransformerModel(VOCAB_SIZE).to(DEVICE)
    print("✅ Blank Transformer Model Initialized.")

    # Train the model
    train_model(model, train_data)

    # Run inference
    print("\n💬 Example Generation:")
    print(generate_text(model, tokenizer, "Once upon a time"))
