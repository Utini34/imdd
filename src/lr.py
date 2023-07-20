from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


class LRSentiment:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_val, y_val):
        predictions = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        f1 = f1_score(y_val, predictions)
        precision = precision_score(y_val, predictions)
        recall = recall_score(y_val, predictions)
        
        print("Logistic Regression Performance")
        print(classification_report(y_val, predictions))
        print("Accuracy: ", accuracy)

        return {
            "model": "lr",
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
