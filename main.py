from video_processing.process_video import process_video
from metrics.evaluation import compute_metrics, plot_confusion_matrix
from report.generate_report import generate_pdf_report
import numpy as np
import os

if __name__ == "__main__":
    # Definición de rutas
    INPUT_VIDEO = "data/input_video.mp4"
    OUTPUT_VIDEO = "output/labeled_video.mp4"
    MODEL_PATH = "yolov8n.pt"
    TRUE_LABELS_PATH = "output/true_labels.npy"
    PREDICTED_LABELS_PATH = "output/predicted_labels.npy"
    CONF_MATRIX_PATH = "output/confusion_matrix.png"
    REPORT_PATH = "output/report.pdf"
    CLASS_NAMES = ["car", "motorcycle", "truck", "bus"]

    try:
        # Paso 1: Procesar el video y generar etiquetas
        print("\n1. Procesando video...")
        process_video(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH)

        # Paso 2: Cargar etiquetas y calcular métricas
        print("\n2. Calculando métricas...")
        true_labels = np.load(TRUE_LABELS_PATH)
        predicted_labels = np.load(PREDICTED_LABELS_PATH)

        # Verificar longitudes
        print(f"Número de etiquetas verdaderas: {len(true_labels)}")
        print(f"Número de etiquetas predichas: {len(predicted_labels)}")

        # Calcular métricas
        metrics = compute_metrics(true_labels, predicted_labels, CLASS_NAMES)
        
        # Mostrar métricas
        print("\nResultados:")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")

        # Generar matriz de confusión
        print("\n3. Generando matriz de confusión...")
        plot_confusion_matrix(
            metrics["confusion_matrix"], 
            CLASS_NAMES, 
            CONF_MATRIX_PATH
        )
        print(f"Matriz de confusión guardada en: {CONF_MATRIX_PATH}")

        # Paso 3: Generar reporte PDF
        print("\n4. Generando reporte PDF...")
        generate_pdf_report(
            metrics=metrics,
            confusion_matrix_path=CONF_MATRIX_PATH,
            output_path=REPORT_PATH
        )

        print("\n¡Proceso completado exitosamente!")
        print(f"- Video procesado: {OUTPUT_VIDEO}")
        print(f"- Matriz de confusión: {CONF_MATRIX_PATH}")
        print(f"- Reporte PDF: {REPORT_PATH}")

    except FileNotFoundError as e:
        print(f"\nError: No se encontró algún archivo necesario: {str(e)}")
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")