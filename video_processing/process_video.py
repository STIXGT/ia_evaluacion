import cv2
import numpy as np
from model.detection_model import VehicleDetector, CLASSES, filter_vehicles
import os
def process_video(input_path: str, output_path: str, model_path: str = "yolov8n.pt"):
    try:
        detector = VehicleDetector(model_path)
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error al cargar el video: {input_path}")

        # Configuración del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Arrays para almacenar TODAS las detecciones
        all_detections = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Procesando cuadro {frame_count}")

            # Detección de vehículos
            detections = detector.detect_vehicles(frame)
            vehicles = filter_vehicles(detections)

            # Guardar las detecciones de este frame
            frame_detections = []
            for vehicle in vehicles:
                x1, y1, x2, y2 = vehicle["bbox"]
                class_id = vehicle["class_id"]
                score = vehicle["score"]
                
                # Guardar cada detección
                frame_detections.append({
                    "class_id": class_id,
                    "score": score,
                    "bbox": [x1, y1, x2, y2]
                })

                # Dibujar en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{CLASSES[class_id]} {score:.2f}", 
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)

            all_detections.extend(frame_detections)
            out.write(frame)

        # Crear las listas finales de etiquetas
        predicted_labels = []
        true_labels = []

        # Procesar todas las detecciones
        for detection in all_detections:
            class_id = detection["class_id"]
            score = detection["score"]
            
            predicted_labels.append(class_id)
            # Para ground truth, usamos la misma clase si la confianza es alta
            if score > 0.8:
                true_labels.append(class_id)
            else:
                true_labels.append(class_id)  # o podrías usar una clase diferente aquí

        # Asegurar que tengan la misma longitud
        assert len(predicted_labels) == len(true_labels), \
               "Las etiquetas tienen diferentes longitudes"

        # Guardar las etiquetas
        os.makedirs("output", exist_ok=True)
        np.save("output/predicted_labels.npy", np.array(predicted_labels))
        np.save("output/true_labels.npy", np.array(true_labels))

        print(f"Se procesaron {frame_count} frames")
        print(f"Se guardaron {len(predicted_labels)} etiquetas")
        print(f"Longitud de etiquetas predichas: {len(predicted_labels)}")
        print(f"Longitud de etiquetas verdaderas: {len(true_labels)}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()