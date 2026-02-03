from ultralytics import YOLO

class YoloSensor:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        pass

    def infer(self, frame):
        raw_result = self.model(frame, verbose=False)
        return self.clear_result(raw_result)

    def clear_result(self, results):
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