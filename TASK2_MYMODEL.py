import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import time

# ======================
# CONFIG
# ======================
TRAIN_CSV = "train.csv"
TEST_LABELS_CSV = "test_labels.csv"  # already merged test + relevance
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
DROPOUT = 0.5
MAX_SEQ_LEN = 50  # max words / tokens
EMBED_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD DATA
# ======================
train_df = pd.read_csv(TRAIN_CSV, encoding="latin1")
test_df = pd.read_csv(TEST_LABELS_CSV, encoding="latin1")  # contains relevance

# ======================
# TOKENIZATION: words/character-combinations
# ======================
def tokenize(text):
    # simple split by space; keeps special chars like "¾", "90°", "#SC"
    return str(text).lower().split()

train_df['search_tokens'] = train_df['search_term'].apply(tokenize)
train_df['title_tokens'] = train_df['product_title'].apply(tokenize)
test_df['search_tokens'] = test_df['search_term'].apply(tokenize)
test_df['title_tokens'] = test_df['product_title'].apply(tokenize)

# ======================
# TRAIN WORD2VEC EMBEDDINGS
# ======================
sentences = list(train_df['search_tokens']) + list(train_df['title_tokens'])
w2v_model = Word2Vec(sentences, vector_size=EMBED_DIM, window=5, min_count=1, workers=4, epochs=10)

# build vocab & embedding matrix
vocab = {word: i+1 for i, word in enumerate(w2v_model.wv.index_to_key)}
vocab['<PAD>'] = 0
embedding_matrix = np.zeros((len(vocab), EMBED_DIM))
for word, idx in vocab.items():
    if word != '<PAD>':
        embedding_matrix[idx] = w2v_model.wv[word]

def encode(tokens):
    return [vocab.get(w, 0) for w in tokens]

def pad(seq, max_len):
    return seq[:max_len] + [0]*(max_len-len(seq))

# ======================
# DATASET
# ======================
class WordDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s = pad(encode(row['search_tokens']), MAX_SEQ_LEN)
        t = pad(encode(row['title_tokens']), MAX_SEQ_LEN)
        y = torch.tensor(row['relevance'], dtype=torch.float32)
        return torch.tensor(s), torch.tensor(t), y

train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)
train_loader = DataLoader(WordDataset(train_split), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(WordDataset(val_split), batch_size=BATCH_SIZE)
test_loader = DataLoader(WordDataset(test_df), batch_size=BATCH_SIZE)

# ======================
# MODEL
# ======================
class WordEncoder(nn.Module):
    def __init__(self, embedding_matrix, out_channels=128, dropout=0.2):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )

        # Multi-kernel CNNs
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, out_channels, kernel_size=k, padding=k // 2)
            for k in [2, 3, 4]  # n-gram sizes
        ])

        self.fc = nn.Linear(len(self.convs) * out_channels, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            c = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # max over sequence
            conv_outs.append(c)
        x = torch.cat(conv_outs, dim=1)  # [batch, out_channels * num_kernels]
        x = self.dropout(x)
        return self.fc(x)


class SiameseWordNet(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        self.encoder = WordEncoder(embedding_matrix,dropout=DROPOUT)
        self.fc = nn.Linear(256, 1)

    def forward(self, s, t):
        es = self.encoder(s)
        et = self.encoder(t)
        x = self.fc(torch.cat([es, et], dim=1))
        x = torch.sigmoid(x) * 2 + 1  # maps 0->1, 1->3
        return x.squeeze(1)  # <-- Option 2: ensures output shape is [batch]

model = SiameseWordNet(embedding_matrix).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ======================
# TRAINING
# ======================
train_rmse_list, val_rmse_list = [], []
train_mae_list, val_mae_list = [], []

print("Training Siamese Word-Level model...")
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

    # Validation
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

    print(f"Epoch {epoch+1} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
          f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

# ======================
# TEST INFERENCE
# ======================
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for s, t, y in test_loader:
        s, t, y = s.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
        pred = model(s, t)
        test_preds.append(pred.cpu())
        test_labels.append(y.cpu())

test_preds_cat = torch.cat(test_preds)
test_labels_cat = torch.cat(test_labels)
test_rmse = np.sqrt(mean_squared_error(test_labels_cat, test_preds_cat))
test_mae = mean_absolute_error(test_labels_cat, test_preds_cat)
runtime = time.time() - start_time

print(f"\nTest RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")
print(f"Total Runtime (Train + Inference): {runtime:.2f} sec")

# ======================
# PLOTS
# ======================
plt.figure(figsize=(12,5))
plt.plot(range(1,EPOCHS+1), train_rmse_list, label="Train RMSE")
plt.plot(range(1,EPOCHS+1), val_rmse_list, label="Val RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Train vs Validation RMSE (Word-Level Siamese)")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(range(1,EPOCHS+1), train_mae_list, label="Train MAE")
plt.plot(range(1,EPOCHS+1), val_mae_list, label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Train vs Validation MAE (Word-Level Siamese)")
plt.legend()
plt.show()
