import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def mse(y_true, y_pred):
    return (f"MSE: {np.mean((y_true - y_pred) ** 2)}")

def mae(y_true, y_pred):
    return (f"MAE: {np.mean(np.abs(y_true - y_pred))}")