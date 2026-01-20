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
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-5
DROPOUT=0.5
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

class CharTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, nhead=4, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(embed_dim, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb(x).permute(1,0,2)  # [seq, batch, embed_dim]
        x = self.transformer(x)         # [seq, batch, embed_dim]
        x = x.permute(1,2,0)            # [batch, embed_dim, seq]
        x = self.pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc(x)))
        return x


class SiameseTransformerRegression(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        nhead=4,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()
        # Our encoder is the CharTransformerEncoder
        self.encoder = CharTransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        # Siamese head: concat outputs from search + title
        self.fc = nn.Linear(128*2, 1)  # 128 from encoder output × 2

    def forward(self, s, t):
        """
        s: [batch, seq_len] search term indices
        t: [batch, seq_len] product title indices
        """
        es = self.encoder(s)  # [batch, 128]
        et = self.encoder(t)  # [batch, 128]
        combined = torch.cat([es, et], dim=1)  # [batch, 256]
        out = self.fc(combined).squeeze(1)     # [batch]
        return out



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
model = SiameseTransformerRegression(
    vocab_size=len(char2idx),
    embed_dim=32,      # embedding size for characters
    nhead=2,           # attention heads
    hidden_dim=64,    # transformer feedforward dimension
    num_layers=2,      # number of transformer encoder layers
    dropout=DROPOUT    # same as your config (0.4)
).to(DEVICE)
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

torch.save(model.state_dict(), "char_siamese_TRANS_model_state.pth")
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
