import numpy as np
import torch
import torch.nn as nn

from transformers import T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import binary_accuracy


class T5Sentiment(nn.Module):
    def __init__(self, pretrained_model):
        super(T5Sentiment, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def train_loop(self, train_loader, valid_loader, optimizer, epochs):
        def train_epoch(model):
            epoch_loss = 0
            correct_count = 0

            model.train()
            for batch in train_loader:
                # Move input and targets to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["target_ids"].to(device)

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                # evalute output tokens
                predicted_value = torch.argmax(outputs.logits, dim=-1)

                correct = (predicted_value == labels).all(axis=1).sum().item()
                correct_count += correct

                epoch_loss += loss.item()

            return epoch_loss / len(train_loader), correct_count / len(train_loader)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(self)
            valid_metrics = self.evaluate_loop(valid_loader)
            print(
                f"Epoch: {epoch + 1} | Train loss: {train_loss:.3f} | Train acc: {train_acc * 100:.2f}% | Validation loss: {valid_metrics['epoch_loss']:.3f} | Validation acc: {valid_metrics['epoch_acc'] * 100:.2f}%"
            )

    def evaluate_loop(self, eval_loader):
        epoch_loss = 0
        correct_count = 0
        # preds = []
        # labels = []
        self.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["target_ids"].to(device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                predicted_value = torch.argmax(outputs.logits, dim=-1)

                loss = outputs.loss
                epoch_loss += loss.item()

                correct = (predicted_value == labels).all(axis=1).sum().item()
                correct_count += correct

        return {
            "model": "t5",
            "accuracy": correct_count / len(eval_loader),
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "epoch_loss": epoch_loss / len(eval_loader),
            "epoch_acc": correct_count / len(eval_loader),
        }
