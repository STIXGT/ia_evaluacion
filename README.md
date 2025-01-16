````
# Detección y Clasificación de Vehículos en Videos

Este proyecto utiliza inteligencia artificial para detectar y clasificar vehículos (como carros, SUVs, camiones y motocicletas) en videos utilizando el modelo YOLOv8. El propósito es analizar un video, etiquetar los vehículos detectados y generar un reporte con las métricas de desempeño.

## Introducción

El objetivo principal de este proyecto es la detección y clasificación de vehículos en videos. A través del uso de un modelo de IA preentrenado, el sistema es capaz de identificar diferentes tipos de vehículos, generar un video etiquetado con cuadros delimitadores, y proporcionar métricas de rendimiento del modelo.

## Metodología

1. **Preparación del entorno**: Se utilizó Python como lenguaje principal, con las siguientes librerías:
    - `opencv-python`: Para el procesamiento de videos.
    - `numpy`: Para manipulación de datos.
    - `matplotlib`: Para la visualización de gráficas.

2. **Modelo de detección**: YOLOv8 (You Only Look Once) fue utilizado para la detección en tiempo real de vehículos en los videos.

3. **Procesamiento del video**: El video es procesado cuadro por cuadro. Para cada cuadro, se realiza la detección de vehículos y se etiquetan con la clase correspondiente.

4. **Evaluación del modelo**: Se calculan métricas como la precisión, el recall y el F1-score, además de la matriz de confusión.

5. **Generación del reporte**: Se genera un PDF con imágenes del video procesado y las gráficas de evaluación.

## Requisitos

Para poder ejecutar el proyecto en tu máquina local, asegúrate de tener Python 3.x instalado y los siguientes paquetes en tu entorno:

```txt
opencv-python
numpy
matplotlib
torch

````

Puedes instalar todos los requerimientos utilizando el siguiente comando:

```
bash
```

CopiarEditar
`pip install -r requirements.txt`
Instrucciones de Instalación y Uso

1. Clona este repositorio a tu máquina local:

```
bash
```

CopiarEditar
`git clone https://github.com/STIXGT/ia_evaluacion.git cd ia_evaluacion`

1. Crea y activa un entorno virtual (opcional pero recomendado):

```
bash
```

CopiarEditar
`python -m venv .venv source .venv/bin/activate # En Linux/macOS .venv\Scripts\activate # En Windows`

1. Instala las dependencias:

```
bash
```

CopiarEditar
`pip install -r requirements.txt`

1. Coloca tu video de entrada en la carpeta `data` y asegúrate de que el archivo tenga el nombre `input_video.mp4`.
2. Ejecuta el script principal para procesar el video:

```
bash
```

CopiarEditar
`python main.py`
Este comando generará un video etiquetado y guardará las métricas de desempeño en la carpeta `output`.
Resultados
El proyecto genera un video procesado en el que se muestran los vehículos detectados con cuadros delimitadores. Además, se calculan métricas de desempeño como precisión, recall, F1-score, y una matriz de confusión.
Contribuciones
Si deseas contribuir al proyecto, por favor abre un "issue" o envía un "pull request" con tus cambios. Asegúrate de seguir las mejores prácticas y de escribir pruebas para cualquier funcionalidad nueva.
Licencia
Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo LICENSE.
Repositorio en GitHub
Puedes acceder al código fuente y otros recursos relacionados en el siguiente enlace:
GitHub - Detección y Clasificación de Vehículos
