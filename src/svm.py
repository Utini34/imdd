from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


class SVMSentiment:
    def __init__(self):
        self.model = SVC(kernel="linear", C=0.025)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_val, y_val):
        val_svm_predictions = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, val_svm_predictions)
        f1 = f1_score(y_val, val_svm_predictions)
        precision = precision_score(y_val, val_svm_predictions)
        recall = recall_score(y_val, val_svm_predictions)

        print("SVM Performance")
        print(classification_report(y_val, val_svm_predictions))
        print("Accuracy: ", accuracy)

        return {
            "model": "svm",
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
