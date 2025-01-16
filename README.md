# Sistema de Detección y Clasificación de Vehículos

Este proyecto utiliza un modelo de inteligencia artificial basado en **YOLOv8** para detectar y clasificar vehículos en videos. Los vehículos detectados se etiquetan en un video de salida, y se generan métricas y reportes detallados para analizar el rendimiento del modelo.

## 🚀 Funcionalidades

- Procesamiento de videos para detectar vehículos (carros, SUVs, camiones, motocicletas, etc.).
- Clasificación de vehículos en tiempo real.
- Generación de un video etiquetado con las detecciones realizadas.
- Cálculo de métricas como precisión, recall y F1-score.
- Creación de un reporte PDF con los resultados del análisis.

## 📁 Estructura del Proyecto

```plaintext
project/
├── main.py                # Punto de entrada del proyecto
├── model/
│   └── detection_model.py # Modelo de IA para detección y clasificación
├── video_processing/
│   └── process_video.py   # Procesamiento del video y generación del video de salida
├── metrics/
│   └── evaluation.py      # Evaluación del modelo: métricas y gráficos
├── report/
│   └── generate_report.py # Generación del reporte PDF
├── data/
│   └── input_video.mp4    # Video de entrada
├── output/
│   ├── labeled_video.mp4  # Video etiquetado generado
│   └── report.pdf         # Reporte PDF generado
└── requirements.txt       # Dependencias del proyecto
```
