import numpy as np
import torch
import torch.nn as nn

from utils import binary_accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class LSTMSentiment(nn.Module):
    def __init__(
        self,
        embedding_tensor,
        embedding_dim,
        hidden_dim,
        output_dim,
        dropout_rate=0.5,
        num_layers=2,
    ):
        super(LSTMSentiment, self).__init__()

        # copy pretrain embedding weight
        self.embedding_dim = embedding_dim
        num_embeddings, embedding_dim = embedding_tensor.shape
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.copy_(embedding_tensor)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, texts):
        embedded = self.dropout(self.embeddings(texts))  # [B,L,N]
        # TODO: add packing and unpacking
        lstm_output, _ = self.lstm(embedded)  # [B,L,D]
        hidden = torch.cat(
            (
                lstm_output[:, -1, : self.embedding_dim],
                lstm_output[:, 0, self.embedding_dim :],
            ),
            dim=1,
        )  # [B,2N], concat forward and reverse
        dense_output = self.fc(hidden)  # [B, O]
        true_label_prob = torch.sigmoid(dense_output)  # [B, O]
        return true_label_prob

    def train_loop(self, train_loader, valid_loader, optimizer, criterion, epochs):
        def train_epoch(model):
            epoch_loss = 0
            epoch_acc = 0

            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                predictions = model(batch["encoded_tokens"]).squeeze(1)
                loss = criterion(predictions, batch["label"].float())
                acc = binary_accuracy(predictions, batch["label"])
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        criterion.to(device)

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(self)
            valid_metrics = self.evaluate_loop(valid_loader, criterion)
            print(
                f"Epoch: {epoch + 1} | Train loss: {train_loss:.3f} | Train acc: {train_acc * 100:.2f}% | Validation loss: {valid_metrics['epoch_loss']:.3f} | Validation acc: {valid_metrics['epoch_acc'] * 100:.2f}%"
            )

    def evaluate_loop(self, eval_loader, criterion):
        epoch_loss = 0
        epoch_acc = 0
        preds = []
        labels = []
        self.eval()
        
        with torch.no_grad():
            for batch in eval_loader:
                predictions = self(batch["encoded_tokens"]).squeeze(1)
                loss = criterion(predictions, batch["label"].float())
                acc = binary_accuracy(predictions, batch["label"])
                epoch_loss += loss.item()
                epoch_acc += acc.item()

                preds.append((predictions >= 0.5).int().tolist())
                labels.append(batch["label"].tolist())

        preds = np.hstack(preds)
        labels = np.hstack(labels)

        return {
            "model": "lstm",
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "epoch_loss": epoch_loss / len(eval_loader),
            "epoch_acc": epoch_acc / len(eval_loader),
        }
