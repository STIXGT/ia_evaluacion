# Detección y Clasificación de Vehículos en Videos

Este proyecto utiliza inteligencia artificial para detectar y clasificar vehículos (como carros, SUVs, camiones y motocicletas) en videos utilizando el modelo YOLOv8. El propósito es analizar un video, etiquetar los vehículos detectados y generar un reporte con las métricas de desempeño.

## Introducción

El objetivo principal de este proyecto es la detección y clasificación de vehículos en videos. A través del uso de un modelo de IA preentrenado, el sistema es capaz de identificar diferentes tipos de vehículos, generar un video etiquetado con cuadros delimitadores, y proporcionar métricas de rendimiento del modelo. El proyecto incluye la creación de un video con las etiquetas de los vehículos detectados y un reporte detallado de las métricas evaluadas.

## Metodología

El proceso de detección y análisis de vehículos en videos consta de los siguientes pasos:

1.  **Preparación del entorno**: Se utilizó Python como lenguaje principal, junto con las siguientes librerías:

    - `opencv-python`: Para el procesamiento de videos e imágenes.
    - `numpy`: Para la manipulación de datos.
    - `matplotlib`: Para la visualización de gráficas y métricas.
    - `torch` y `torchvision`: Para cargar y ejecutar el modelo YOLOv8 preentrenado.

2.  **Modelo de detección**: YOLOv8 (You Only Look Once) fue utilizado para realizar la detección en tiempo real de vehículos en los videos. Este modelo es conocido por su velocidad y precisión en tareas de visión por computadora.

3.  **Procesamiento del video**: El video de entrada se procesa cuadro por cuadro. Para cada cuadro, se realiza la detección de vehículos y se etiquetan con la clase correspondiente (carro, SUV, camión, motocicleta).

4.  **Evaluación del modelo**: Se calculan métricas como la **precisión**, **recall**, **F1-score**, y la **matriz de confusión** para evaluar el desempeño del modelo de detección.

5.  **Generación del reporte**: Se genera un archivo PDF que contiene imágenes del video procesado, gráficas de evaluación de las métricas y análisis del rendimiento del modelo.

## Requisitos

Para ejecutar este proyecto en tu máquina local, asegúrate de tener Python 3.x instalado y las siguientes dependencias:

```txt
opencv-python==4.6.0.66
numpy==1.24.0
matplotlib==3.6.2
torch==1.13.1
torchvision==0.14.1
pyyaml==6.0
tqdm==4.64.1
reportlab==3.6.12
scikit-learn==1.1.3
```

Puedes instalar todas las dependencias con el siguiente comando:

```bash
pip install -r requirements.txt
```

## Instrucciones de Instalación y Uso

1.  Clona este repositorio a tu máquina local:

```bash
git clone https://github.com/STIXGT/ia_evaluacion.git
cd ia_evaluacion
```

2.  Crea y activa un entorno virtual (opcional pero recomendado):

```bash
python -m venv .venv
source .venv/bin/activate   # En Linux/macOS
.venv\Scripts\activate      # En Windows
```

3.  Instala las dependencias:

```bash
pip install -r requirements.txt
```

4.  Coloca tu video de entrada en la carpeta `data` y asegúrate de que el archivo tenga el nombre `input_video.mp4`.

5.  Ejecuta el script principal para procesar el video:

```bash
python main.py
```

Este comando generará un video etiquetado en la carpeta `output` y almacenará las métricas de desempeño en archivos `.npy`.

## Resultados

El proyecto genera un video procesado en el que se muestran los vehículos detectados con cuadros delimitadores. Además, se calculan las siguientes métricas de desempeño:

- **Precisión**: Mide la exactitud del modelo al predecir vehículos correctamente.
- **Recall**: Evalúa la capacidad del modelo para detectar todos los vehículos presentes.
- **F1-score**: Una combinación de precisión y recall para evaluar el desempeño general.
- **Matriz de confusión**: Muestra el rendimiento del modelo al comparar las predicciones con las etiquetas reales.
