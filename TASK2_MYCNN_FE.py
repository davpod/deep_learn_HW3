import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import time

# ======================
# CONFIG
# ======================
TRAIN_CSV = "train.csv"
TEST_LABELS_CSV = "test_labels.csv"
BATCH_SIZE = 128
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT = 0.2
MAX_SEARCH_LEN = 30
MAX_TITLE_LEN = 50

# ======================
# LOAD DATA
# ======================
train_df = pd.read_csv(TRAIN_CSV, encoding="latin1")
test_df = pd.read_csv(TEST_LABELS_CSV, encoding="latin1")


# ======================
# CHAR VOCAB (reuse your previous code)
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


def encode(text):
    return [char2idx[c] for c in str(text).lower()]


def pad(seq, max_len):
    return seq[:max_len] + [0] * max(0, max_len - len(seq))


# ======================
# DATASET
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
class CharEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.c1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb(x).permute(0, 2, 1)
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = self.dropout(self.fc(x))
        return x


class SiameseRegression(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CharEncoder(vocab_size, dropout=DROPOUT)
        self.fc = nn.Linear(256, 1)

    def forward(self, s, t):
        es = self.encoder(s)
        et = self.encoder(t)
        return self.fc(torch.cat([es, et], dim=1)).squeeze(1)


# ======================
# LOAD PRETRAINED MODEL
# ======================
feature_model = SiameseRegression(len(char2idx)).to(DEVICE)
feature_model.load_state_dict(torch.load("char_siamese_model_state.pth"))
feature_model.eval()  # freeze by default, but let's explicitly freeze
for param in feature_model.encoder.parameters():
    param.requires_grad = False

# ======================
# CREATE DATALOADERS
# ======================
train_df_split, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
train_loader = DataLoader(SearchDataset(train_df_split), batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(SearchDataset(val_df), batch_size=BATCH_SIZE)
test_loader = DataLoader(SearchDataset(test_df), batch_size=BATCH_SIZE)


# ======================
# 1️⃣ Option: Use embeddings + new FC layer (still DL)
# ======================
class FCHeadRegression(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)


fc_model = FCHeadRegression(input_dim=256).to(DEVICE)
optimizer = torch.optim.Adam(fc_model.parameters(), lr=LR)
criterion = nn.MSELoss()


# --- Extract embeddings function ---
def extract_embeddings(loader):
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for s, t, y in loader:
            s, t = s.to(DEVICE), t.to(DEVICE)
            es = feature_model.encoder(s)
            et = feature_model.encoder(t)
            emb = torch.cat([es, et], dim=1)
            all_embeddings.append(emb.cpu())
            all_labels.append(y)
    return torch.cat(all_embeddings), torch.cat(all_labels)


# Extract train/val embeddings
train_emb, train_labels = extract_embeddings(train_loader)
val_emb, val_labels = extract_embeddings(val_loader)
test_emb, test_labels = extract_embeddings(test_loader)

# --- Train new FC on embeddings ---
start_time=time.time()
EPOCHS_FC = 10
for epoch in range(EPOCHS_FC):
    fc_model.train()
    optimizer.zero_grad()
    outputs = fc_model(train_emb.to(DEVICE))
    loss = criterion(outputs, train_labels.to(DEVICE))
    loss.backward()
    optimizer.step()

    # Validation
    fc_model.eval()
    with torch.no_grad():
        val_outputs = fc_model(val_emb.to(DEVICE))
        train_rmse = np.sqrt(mean_squared_error(train_labels, outputs.cpu()))
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_outputs.cpu()))
    print(f"Epoch {epoch + 1} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

# ======================
# 2️⃣ Option: Classical ML (Ridge)
# ======================
# ======================
# 2️⃣ Option: Classical ML (Ridge)
# ======================
start_time_ridge = time.time()
ridge = Ridge(alpha=1.0)
ridge.fit(train_emb.numpy(), train_labels.numpy())

# --- Training metrics ---
train_preds_ridge = ridge.predict(train_emb.numpy())
train_rmse_ridge = np.sqrt(mean_squared_error(train_labels.numpy(), train_preds_ridge))
train_mae_ridge = mean_absolute_error(train_labels.numpy(), train_preds_ridge)

# --- Validation metrics ---
val_preds_ridge = ridge.predict(val_emb.numpy())
val_rmse_ridge = np.sqrt(mean_squared_error(val_labels.numpy(), val_preds_ridge))
val_mae_ridge = mean_absolute_error(val_labels.numpy(), val_preds_ridge)

# --- Test metrics ---
test_preds_ridge = ridge.predict(test_emb.numpy())
test_rmse_ridge = np.sqrt(mean_squared_error(test_labels.numpy(), test_preds_ridge))
test_mae_ridge = mean_absolute_error(test_labels.numpy(), test_preds_ridge)

ridge_runtime = time.time() - start_time_ridge

print(f"\nRidge Train RMSE: {train_rmse_ridge:.4f} | MAE: {train_mae_ridge:.4f}")
print(f"Ridge Val RMSE:   {val_rmse_ridge:.4f} | MAE: {val_mae_ridge:.4f}")
print(f"Ridge Test RMSE:  {test_rmse_ridge:.4f} | MAE: {test_mae_ridge:.4f}")
print(f"Ridge Runtime: {ridge_runtime:.2f} sec")
