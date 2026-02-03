import cv2
import time

from yolo_sensor import YoloSensor
from recorder import Recorder
from inference import Inference, InferenceMode
from camera import CameraThread

class Watcher:
    def __init__(self, mode: int, show_display=True, use_threading=False):
        self.detector = YoloSensor()
        self.recorder = Recorder()

        self.inference = Inference()
        self.inference.mode = InferenceMode(mode)

        self.use_threading = use_threading

        self.cam = 0
        self.show_display = show_display
        self.cap = None
        self.is_running = False

    def start(self):
        if self.use_threading:
            print("🚀 Iniciando com CameraThread (Buffer Automático)...")
            self.cap = CameraThread(self.cam).start()
            time.sleep(1.0)  # Aquece a thread
        else:
            print("📷 Iniciando com OpenCV Padrão (Single-thread)...")
            self.cap = cv2.VideoCapture(self.cam)

        infra_name = "THREADED" if self.use_threading else "STANDARD"

        self.recorder.start()
        self.is_running = True
        frame_index = 0
        start_time = time.time()
        was_blinking = False

        last_detections = []

        try:
            current_mode_name = self.inference.mode.name
        except AttributeError:
            current_mode_name = str(self.inference.mode)

        infra_type = "THREADED" if self.use_threading else "STANDARD"
        print(f"Watcher iniciado: Modo {current_mode_name} | Infra: {infra_type}")

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            now = time.time()
            frame_index += 1

            can_infer = self.inference.can_infer(now, frame, last_detections)

            if not self.use_threading:
                if can_infer and was_blinking:
                    # Limpeza manual para OpenCV padrão
                    for _ in range(4): self.cap.grab()
                    ret, frame = self.cap.read()
                    if not ret: break

            was_blinking = not can_infer

            if can_infer:
                t_start = time.perf_counter()
                frame, detections = self.detector.infer_draw(frame)
                last_detections = detections

                t_end = time.perf_counter()

                infer_time_ms = (t_end - t_start) * 1000

                self.recorder.record(frame_index, infer_time_ms, detections, current_mode_name,infra_name)
            else:
                self.recorder.record(frame_index, 0, [], current_mode_name,infra_name)
                last_detections = []
                time.sleep(self.inference.get_sleep_time())

            if self.show_display:
                cv2.imshow("Watcher feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False

            elapsed_time = time.time() - start_time
            if elapsed_time > 30:
                self.is_running = False
                print("Tempo limite de benchmark (30s) excedido.")

        if self.use_threading:
            self.cap.release()
        else:
            self.cap.release()

        self.recorder.stop()
        self.recorder.save_csv("data/benchmark.csv")
        self.recorder.save_env_info()
        print(self.recorder.summary())
        cv2.destroyAllWindows()