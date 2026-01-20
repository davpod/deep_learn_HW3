import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
TRAIN_CSV = "train.csv"
TEST_LABELS_CSV = "test_labels.csv"  # already merged test + relevance
BATCH_SIZE = 512
EPOCHS = 15
LR = 1e-3
DROPOUT=0.4
MAX_SEARCH_LEN = 30
MAX_TITLE_LEN = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD DATA
# ======================
train_df = pd.read_csv(TRAIN_CSV, encoding="latin1")
test_df = pd.read_csv(TEST_LABELS_CSV, encoding="latin1")  # contains relevance

# ======================
# CHAR VOCAB
# ======================
def build_char_vocab(dfs):
    chars = set()
    for df in dfs:
        for col in ["search_term", "product_title"]:
            for text in df[col].astype(str):
                chars.update(text.lower())
    char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
    char2idx["<PAD>"] = 0
    return char2idx

char2idx = build_char_vocab([train_df, test_df])
print("total chars:", len(char2idx)-1)
def encode(text):
    return [char2idx[c] for c in str(text).lower()]

def pad(seq, max_len):
    return seq[:max_len] + [0] * max(0, max_len - len(seq))

# ======================
# DATASETS
# ======================
class SearchDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s = pad(encode(row["search_term"]), MAX_SEARCH_LEN)
        t = pad(encode(row["product_title"]), MAX_TITLE_LEN)
        y = torch.tensor(row["relevance"], dtype=torch.float32)
        return torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long), y

# ======================
# MODELS
# ======================

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: [B, L, 2H]
        scores = self.attn(x).squeeze(-1)       # [B, L]
        weights = torch.softmax(scores, dim=1) # [B, L]
        return (x * weights.unsqueeze(-1)).sum(dim=1)

class CharBiLSTMAttnEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.attn = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb(x)            # [B, L, E]
        x, _ = self.lstm(x)        # [B, L, 2H]
        x = self.attn(x)           # [B, 2H]
        x = self.dropout(self.fc(x))
        return x

class CharBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb(x)                  # [B, L, E]
        _, (h_n, _) = self.lstm(x)       # h_n: [2, B, H]
        h = torch.cat([h_n[0], h_n[1]], dim=1)  # [B, 2H]
        h = self.dropout(self.fc(h))
        return h

class SiameseRegression(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CharBiLSTMEncoder(
            vocab_size,
            embed_dim=32,
            hidden_dim=64,
            dropout=DROPOUT
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, s, t):
        es = self.encoder(s)
        et = self.encoder(t)
        return self.fc(torch.cat([es, et], 1)).squeeze(1)

# class SiameseRegression(nn.Module):
#     def __init__(self, vocab_size):
#         super().__init__()
#         self.encoder = CharBiLSTMAttnEncoder(vocab_size)
#         self.fc = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )
#
#     def forward(self, s, t):
#         es = self.encoder(s)
#         et = self.encoder(t)
#         return self.fc(torch.cat([es, et], 1)).squeeze(1)

# ======================
# TRAIN / VALID SPLIT
# ======================
train_df_split, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
train_loader = DataLoader(SearchDataset(train_df_split), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SearchDataset(val_df), batch_size=BATCH_SIZE)
test_loader = DataLoader(SearchDataset(test_df), batch_size=BATCH_SIZE)

# ======================
# TRAIN NEURAL MODEL WITH TIMING AND LOGGING
# ======================
model = SiameseRegression(len(char2idx)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Lists to save metrics per epoch for plots
train_rmse_list, train_mae_list = [], []
val_rmse_list, val_mae_list = [], []

print("Training Siamese model...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_preds, train_labels = [], []

    for s, t, y in train_loader:
        s, t, y = s.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(s, t)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_preds.append(pred.detach().cpu())
        train_labels.append(y.cpu())

    train_preds_cat = torch.cat(train_preds)
    train_labels_cat = torch.cat(train_labels)
    train_rmse = np.sqrt(mean_squared_error(train_labels_cat, train_preds_cat))
    train_mae = mean_absolute_error(train_labels_cat, train_preds_cat)
    train_rmse_list.append(train_rmse)
    train_mae_list.append(train_mae)
    print(f"Epoch {epoch+1} | Train RMSE: {train_rmse:.4f} | Train MAE: {train_mae:.4f}")

    # Validation after each epoch
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for s, t, y in val_loader:
            s, t, y = s.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
            pred = model(s, t)
            val_preds.append(pred.cpu())
            val_labels.append(y.cpu())
    val_preds_cat = torch.cat(val_preds)
    val_labels_cat = torch.cat(val_labels)
    val_rmse = np.sqrt(mean_squared_error(val_labels_cat, val_preds_cat))
    val_mae = mean_absolute_error(val_labels_cat, val_preds_cat)
    val_rmse_list.append(val_rmse)
    val_mae_list.append(val_mae)
    print(f"Epoch {epoch+1} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}")

# ======================
# NEURAL TEST INFERENCE
# ======================
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for s, t, y in test_loader:
        s, t, y = s.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
        pred = model(s, t)
        test_preds.append(pred.cpu())
        test_labels.append(y.cpu())

test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)
test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
test_mae = mean_absolute_error(test_labels, test_preds)
nn_runtime = time.time() - start_time
print(f"\nFINAL TEST RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")
print(f"Total Neural Model Runtime (Train + Inference): {nn_runtime:.2f} sec")

torch.save(model.state_dict(), "char_siamese_BILTSM_model_state.pth")
print("Model weights saved successfully!")
# ======================
# PLOT TRAINING CURVES
# ======================
plt.figure(figsize=(10,5))
plt.plot(range(1,EPOCHS+1), train_rmse_list, label="Train RMSE")
plt.plot(range(1,EPOCHS+1), val_rmse_list, label="Val RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Character-level Siamese RMSE over epochs")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(1,EPOCHS+1), train_mae_list, label="Train MAE")
plt.plot(range(1,EPOCHS+1), val_mae_list, label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Character-level Siamese MAE over epochs")
plt.legend()
plt.grid()
plt.show()
