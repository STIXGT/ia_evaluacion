from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Inicializa el modelo YOLOv8 para la detección de vehículos.
        :param model_path: Ruta al modelo YOLOv8 preentrenado.
        """
        self.model = YOLO(model_path)

    def detect_vehicles(self, frame):
        """
        Detecta vehículos en un cuadro de video.
        :param frame: Imagen en formato numpy array (BGR).
        :return: Resultados del modelo y cuadros delimitadores con clases.
        """
        results = self.model(frame)
        detections = []

        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class_id": int(class_id)
            })

        return detections

# Clases predefinidas (según el modelo COCO)
CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def filter_vehicles(detections):
    """
    Filtra detecciones para incluir solo vehículos de interés.
    :param detections: Lista de detecciones del modelo.
    :return: Lista de vehículos filtrados.
    """
    return [d for d in detections if d["class_id"] in CLASSES]
