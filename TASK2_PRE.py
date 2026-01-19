import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import AutoTokenizer, AutoModel
import time
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
TRAIN_CSV = "train.csv"
TEST_LABELS_CSV = "test_labels.csv"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-uncased"  # pre-trained transformer

# ======================
# LOAD DATA
# ======================
train_df = pd.read_csv(TRAIN_CSV, encoding="latin1")
test_df = pd.read_csv(TEST_LABELS_CSV, encoding="latin1")  # contains relevance

# ======================
# DATASET
# ======================
class SentenceEmbeddingDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["search_term"]) + " || " + str(row["product_title"])
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        label = torch.tensor(row["relevance"], dtype=torch.float32)
        return input_ids, attention_mask, label

# ======================
# MODEL (BERT frozen, only regression head trainable)
# ======================
class BertRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # freeze BERT
        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(hidden_size, 1)  # trainable only

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # ensure no gradients for BERT
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.fc(cls_emb).squeeze(1)

# ======================
# TOKENIZER & DATALOADERS
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)

train_dataset = SentenceEmbeddingDataset(train_split, tokenizer)
val_dataset = SentenceEmbeddingDataset(val_split, tokenizer)
test_dataset = SentenceEmbeddingDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================
# TRAINING LOOP
# ======================
model = BertRegression(MODEL_NAME).to(DEVICE)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)  # only train regression head
criterion = nn.MSELoss()

print("Fine-tuning regression head on frozen BERT...")
start_time = time.time()

train_rmse_list, val_rmse_list = [], []
train_mae_list, val_mae_list = [], []

for epoch in range(EPOCHS):
    # ----- Training -----
    model.train()
    train_preds, train_labels = [], []
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_preds.append(outputs.detach().cpu())
        train_labels.append(labels.cpu())

    train_preds_cat = torch.cat(train_preds)
    train_labels_cat = torch.cat(train_labels)
    train_rmse = np.sqrt(mean_squared_error(train_labels_cat, train_preds_cat))
    train_mae = mean_absolute_error(train_labels_cat, train_preds_cat)
    train_rmse_list.append(train_rmse)
    train_mae_list.append(train_mae)

    # ----- Validation -----
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            outputs = model(input_ids, attention_mask)
            val_preds.append(outputs.cpu())
            val_labels.append(labels.cpu())
    val_preds_cat = torch.cat(val_preds)
    val_labels_cat = torch.cat(val_labels)
    val_rmse = np.sqrt(mean_squared_error(val_labels_cat, val_preds_cat))
    val_mae = mean_absolute_error(val_labels_cat, val_preds_cat)
    val_rmse_list.append(val_rmse)
    val_mae_list.append(val_mae)

    print(f"Epoch {epoch+1} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

# ======================
# TEST INFERENCE
# ======================
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        outputs = model(input_ids, attention_mask)
        test_preds.append(outputs.cpu())
        test_labels.append(labels.cpu())

test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)
test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
test_mae = mean_absolute_error(test_labels, test_preds)

runtime = time.time() - start_time
print(f"\nTest RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")
print(f"Total Runtime (Train + Inference): {runtime:.2f} sec")

# ======================
# PLOTTING
# ======================
plt.figure(figsize=(10, 4))
plt.plot(range(1, EPOCHS + 1), train_rmse_list, label="Train RMSE")
plt.plot(range(1, EPOCHS + 1), val_rmse_list, label="Val RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Frozen BERT + Regression Head Training")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(range(1, EPOCHS + 1), train_mae_list, label="Train MAE")
plt.plot(range(1, EPOCHS + 1), val_mae_list, label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Frozen BERT + Regression Head Training")
plt.legend()
plt.show()
