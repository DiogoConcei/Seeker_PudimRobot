import cv2
import time

from recorder import Recorder
from yolo_sensor import YoloSensor


class Watcher:
    def __init__(self, cam_index: int = 0):
        self.detector = YoloSensor()
        self.benchmark = Recorder()

        self.cam_index = cam_index
        self.cap = None
        self.frame_index = 0

        # True -> Pisca / False -> Não pisca
        self.blink_mode  = False
        self.blink_interval = 1.5  # tempo entre piscadas
        self.blink_duration = 0.5  # tempo da piscada
        self.last_blink_time = 0.0
        self.blink_start = 0.0
        self.blinking = False

    def start(self):
        self.cap = cv2.VideoCapture(self.cam_index)

        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível abrir a câmera.")

        print("Watcher iniciado. Pressione ESC para sair.")

        self.benchmark.start()

        try:
            self._watch_loop()
        finally:
            self._cleanup()

    def _watch_loop(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Falha ao capturar frame.")
                break

            self.frame_index += 1
            now = time.perf_counter()

            # -------------------------------
            # MODO DE DESEMPENHO (PISCADA)
            # -------------------------------
            if self.blink_mode:
                self._update_blink_state(now)

                if self.blinking:
                    cv2.imshow("YOLO Watcher", frame)

                    if cv2.waitKey(1) & 0xFF == 27:
                        print("Encerrando watcher.")
                        break

                    continue  # pula para o próximo frame
            # -------------------------------

            # ---- Benchmark da inferência ----
            t0 = time.perf_counter()
            detections = self.detector.infer(frame)
            t1 = time.perf_counter()

            infer_time_ms = (t1 - t0) * 1000
            # --------------------------------

            # Registra benchmark (somente quando houve inferência)
            self.benchmark.record(
                frame_index=self.frame_index,
                infer_time_ms=infer_time_ms,
                num_detections=len(detections)
            )

            # (opcional) visualizar deteções
            if detections:
                print(detections)

            cv2.imshow("YOLO Watcher", frame)

            # ESC para sair
            if cv2.waitKey(1) & 0xFF == 27:
                print("Encerrando watcher.")
                break

    def _cleanup(self):
        print("Finalizando watcher...")

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        self.benchmark.stop()
        self.benchmark.save_csv("data/benchmark.csv")

        summary = self.benchmark.summary()
        print("\n=== RESUMO DO BENCHMARK ===")
        for k, v in summary.items():
            print(f"{k}: {v}")

    def _update_blink_state(self, now: float):
        if not self.blinking:
            if now - self.last_blink_time >= self.blink_interval:
                self.blinking = True
                self.blink_start = now
        else:
            if now - self.blink_start >= self.blink_duration:
                self.blinking = False
                self.last_blink_time = now
