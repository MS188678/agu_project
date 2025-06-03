script.ipynb# Standard library
import os
import re
import math
from collections import Counter

# Data handling
import pandas as pd
import matplotlib.pyplot as plt

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Progress bar
from tqdm import tqdm

# External dataset handling
import kagglehub

EPOCHS = 10

print("Dataset loading...")
path = kagglehub.dataset_download("cosmos98/twitter-and-reddit-sentimental-analysis-dataset")
print("Dataset saved:", path)

csv_file = os.path.join(path, "Twitter_Data.csv")
assert os.path.exists(csv_file), "File not found!"

df = pd.read_csv(csv_file)
df = df.dropna()
print("Examples:")
print(df.head())



def map_label(label):
    return { -1: "negative", 0: "neutral", 1: "positive" }.get(label, "neutral")

df["label"] = df["category"].apply(map_label)

MAX_LEN = 64

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

counter = Counter()
for text in df["clean_text"]:
    counter.update(tokenize(text))

vocab = ["<PAD>", "<UNK>"] + [word for word, freq in counter.items() if freq >= 3]
word2idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(word2idx)
print(f"Dict. size: {vocab_size}")

def encode(text):
    tokens = tokenize(text)
    idxs = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
    idxs = idxs[:MAX_LEN] + [word2idx["<PAD>"]] * (MAX_LEN - len(idxs))
    return idxs

label2class = {"negative": 0, "neutral": 1, "positive": 2}

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["clean_text"], df["label"], test_size=0.05, random_state=42, stratify=df["label"]
)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.X = [encode(text) for text in texts]
        self.y = [label2class[label] for label in labels]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

BATCH_SIZE = 64

train_dataset = SentimentDataset(train_texts, train_labels)
test_dataset = SentimentDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.dropout(self.out_linear(attn_output))

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, mask)))
        return self.norm2(x + self.dropout(self.ff(x)))

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, ff_dim=128, num_layers=2, num_classes=3, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(num_layers, d_model, n_heads, ff_dim, dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, mask)
        x = x.mean(dim=1)
        return self.fc(self.dropout(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = TransformerSentimentClassifier(vocab_size=vocab_size).to(device)

READ_MODEL = True
MODEL_PATH = "/content/transformer_sentiment_model.pt"

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
test_accuracies = []

if READ_MODEL and os.path.exists(MODEL_PATH):
    print("Loading model...")
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    print("Model training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = torch.argmax(model(X_batch), dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        acc = correct / total
        test_accuracies.append(acc)
        print(f"📈 Test accuracy: {acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label="Test Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = torch.argmax(model(X_batch), dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y_batch.tolist())

print("\n Classification report:")
print(classification_report(all_labels, all_preds, target_names=["negative", "neutral", "positive"]))

custom_sentences = [
    "I absolutely loved the experience!",
    "That was the best movie I've seen this year.",
    "The customer service was outstanding.",
    "I would definitely recommend this to my friends.",
    "Everything worked perfectly, thank you!",
    "Such a pleasant surprise!",
    "This product exceeded my expectations.",
    "I'm really happy with how things turned out.",
    "Amazing quality and fast shipping!",
    "The staff were super friendly and helpful.",

    "This is the worst thing I've ever bought.",
    "Completely disappointed with the outcome.",
    "The experience was a total letdown.",
    "It broke after one use — horrible.",
    "Terrible customer support, nobody responded.",
    "Totally not worth the money.",
    "I'm so frustrated with this situation.",
    "Unacceptable behavior from the staff.",
    "Nothing worked as expected.",
    "I regret wasting my time on this.",

    "I'm still waiting to see how it turns out.",
    "It was okay, nothing special.",
    "I don't have strong feelings about this.",
    "This might be useful for some people.",
    "The process was straightforward.",
    "I'll need more time to form an opinion.",
    "Not good, not bad — just average.",
    "It functions as described.",
    "I tried it once, haven't used it since.",
    "I guess it's fine for the price.",
]

print("\n Custom tests:")
with torch.no_grad():
    for sentence in custom_sentences:
        encoded = torch.tensor([encode(sentence)], dtype=torch.long).to(device)
        output = model(encoded)
        pred_class = torch.argmax(output, dim=1).item()
        print(f"📝 '{sentence}' → Predicted: {['negative', 'neutral', 'positive'][pred_class]}")
