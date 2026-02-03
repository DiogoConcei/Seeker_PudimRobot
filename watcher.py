import cv2
from yolo_sensor import YoloSensor

class Watcher:
    def __init__(self):
        self.detector = YoloSensor()
        self.cam = 0
        self.cap = None
        self.running = False

    def start(self):
        self.cap = cv2.VideoCapture(self.cam)

        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível abrir a câmera.")

        self.watching()

        print("Watcher iniciado. Pressione ESC para sair.")
        self.running = True

        try:
            self.watching()
        finally:
            self.cleanup()

    def watching(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Falha ao capturar frame.")
                break

            detections = self.detector.infer(frame)

            if detections:
                print(detections)

            cv2.imshow("YOLO watcher", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("Encerrando watcher.")
                break

    def cleanup(self):
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        print("Watcher finalizado")
