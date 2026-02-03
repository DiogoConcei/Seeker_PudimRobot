from ultralytics import YOLO
import cv2

class YoloSensor:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    #   retorna apenas dados
    def infer_coords(self, frame):
        """
        Retorna:
        (class_id, confidence, x1, y1, x2, y2)
        """
        raw_result = self.model(frame, verbose=False)
        return self._parse_results(raw_result)

    #   retorna frame com boxes desenhadas e lista de detecções
    def infer_draw(self, frame):
        raw_result = self.model(frame, verbose=False)
        detections = self._parse_results(raw_result)

        for det in detections:
            class_id, score, x1, y1, x2, y2 = det
            label = str(class_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame, detections

    def _parse_results(self, results):
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])

                detections.append((
                    class_id,
                    round(confidence, 3),
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ))

        return tuple(detections)