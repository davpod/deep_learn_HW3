import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import time

# ======================
# CONFIG
# ======================
TRAIN_CSV = "train.csv"
TEST_LABELS_CSV = "test_labels.csv"
BATCH_SIZE = 128
LR = 5e-5
EPOCHS_FC = 10
DROPOUT = 0.4
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
    char2idx = {c: i+1 for i, c in enumerate(sorted(chars))}
    char2idx["<PAD>"] = 0
    return char2idx

char2idx = build_char_vocab([train_df, test_df])
print("Total chars:", len(char2idx)-1)

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
# BiLSTM Siamese (same as before)
# ======================
class CharBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb(x)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        h = self.dropout(self.fc(h))
        return h

class SiameseRegression(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CharBiLSTMEncoder(vocab_size, dropout=DROPOUT)
        self.fc = nn.Linear(128*2, 1)

    def forward(self, s, t):
        es = self.encoder(s)
        et = self.encoder(t)
        return self.fc(torch.cat([es, et], dim=1)).squeeze(1)

# ======================
# LOAD PRETRAINED BILTSM
# ======================
feature_model = SiameseRegression(len(char2idx)).to(DEVICE)
feature_model.load_state_dict(torch.load("char_siamese_BILTSM_model_state.pth"))
feature_model.eval()
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
# EXTRACT EMBEDDINGS
# ======================
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

train_emb, train_labels = extract_embeddings(train_loader)
val_emb, val_labels = extract_embeddings(val_loader)
test_emb, test_labels = extract_embeddings(test_loader)

# ======================
# OPTION 1: Small FC HEAD
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

start_time_fc = time.time()
for epoch in range(EPOCHS_FC):
    fc_model.train()
    optimizer.zero_grad()
    outputs = fc_model(train_emb.to(DEVICE))
    loss = criterion(outputs, train_labels.to(DEVICE))
    loss.backward()
    optimizer.step()

    # validation
    fc_model.eval()
    with torch.no_grad():
        val_outputs = fc_model(val_emb.to(DEVICE))
        train_rmse = np.sqrt(mean_squared_error(train_labels, outputs.cpu()))
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_outputs.cpu()))
    print(f"FC Epoch {epoch+1} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")
fc_runtime = time.time() - start_time_fc

# Final evaluation
fc_model.eval()
with torch.no_grad():
    test_outputs = fc_model(test_emb.to(DEVICE))
    test_rmse = np.sqrt(mean_squared_error(test_labels, test_outputs.cpu()))
    test_mae = mean_absolute_error(test_labels, test_outputs.cpu())
print(f"\nFC Head | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f} | Runtime: {fc_runtime:.2f}s")

# ======================
# OPTION 2: Classical ML (Ridge)
# ======================
start_time_ridge = time.time()
ridge = Ridge(alpha=1.0)
ridge.fit(train_emb.numpy(), train_labels.numpy())
pred_test_ridge = ridge.predict(test_emb.numpy())
ridge_rmse = np.sqrt(mean_squared_error(test_labels.numpy(), pred_test_ridge))
ridge_mae = mean_absolute_error(test_labels.numpy(), pred_test_ridge)
ridge_runtime = time.time() - start_time_ridge
print(f"\nRidge | Test RMSE: {ridge_rmse:.4f} | Test MAE: {ridge_mae:.4f} | Runtime: {ridge_runtime:.2f}s")

start_time_rf = time.time()
rf = RandomForestRegressor(
    n_estimators=200,  # number of trees
    max_depth=10,      # limit tree depth to prevent overfitting
    random_state=42,
    n_jobs=-1
)
rf.fit(train_emb.numpy(), train_labels.numpy())
pred_rf = rf.predict(test_emb.numpy())
rf_rmse = np.sqrt(mean_squared_error(test_labels.numpy(), pred_rf))
rf_mae = mean_absolute_error(test_labels.numpy(), pred_rf)
rf_runtime = time.time() - start_time_rf
print(f"Random Forest Test RMSE: {rf_rmse:.4f} | MAE: {rf_mae:.4f} | Runtime: {rf_runtime:.2f}s")

# --- Plain Linear Regression ---
start_time_lr = time.time()
lr = LinearRegression()
lr.fit(train_emb.numpy(), train_labels.numpy())
pred_lr = lr.predict(test_emb.numpy())
lr_rmse = np.sqrt(mean_squared_error(test_labels.numpy(), pred_lr))
lr_mae = mean_absolute_error(test_labels.numpy(), pred_lr)
lr_runtime = time.time() - start_time_lr
print(f"Linear Regression Test RMSE: {lr_rmse:.4f} | MAE: {lr_mae:.4f} | Runtime: {lr_runtime:.2f}s")