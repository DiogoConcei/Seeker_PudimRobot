import cv2
import time

from yolo_sensor import YoloSensor
from recorder import Recorder

class Watcher:
    def __init__(self, mode: int, show_display=True):
        self.detector = YoloSensor()
        self.recorder = Recorder()
        self.mode = mode
        self.cam = 0
        self.show_display = show_display
        self.cap = None
        self.is_running = False
        pass

    def start(self):
        self.cap = cv2.VideoCapture(self.cam)
        self.recorder.start()


        self.is_running = True
        frame_index = 0
        start_time = time.time()

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            t_start = time.perf_counter()
            frame, detections = self.detector.infer_draw(frame)
            t_end = time.perf_counter()

            infer_time_ms = (t_end - t_start) * 1000

            self.recorder.record(frame_index, infer_time_ms, detections)

            if self.show_display:
                cv2.imshow("Watcher feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False

            elapsed_time = time.time() - start_time

            if elapsed_time > 30:
                self.is_running = False
                print("Templo limite excedido")

        self.recorder.stop()
        self.recorder.save_csv("data/benchmark.csv")
        print(self.recorder.summary())