import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def compute_metrics(true_labels, predicted_labels, class_names):
    """
    Calcula métricas de desempeño del modelo.
    :param true_labels: Lista de etiquetas reales.
    :param predicted_labels: Lista de etiquetas predichas.
    :param class_names: Nombres de las clases.
    :return: Diccionario con precisión, recall y F1-score.
    """
    precision = precision_score(true_labels, predicted_labels, average="weighted")
    recall = recall_score(true_labels, predicted_labels, average="weighted")
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm, class_names, output_path="output/confusion_matrix.png"):
    """
    Genera y guarda una matriz de confusión como imagen.
    :param cm: Matriz de confusión.
    :param class_names: Nombres de las clases.
    :param output_path: Ruta para guardar la matriz de confusión.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(output_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {output_path}")
