import cv2
import time

from yolo_sensor import YoloSensor
from recorder import Recorder
from inference import Inference, InferenceMode


class Watcher:
    def __init__(self, mode: int, show_display=True):
        self.detector = YoloSensor()
        self.recorder = Recorder()

        # Inicializa a inferência e define o modo
        self.inference = Inference()
        self.inference.mode = InferenceMode(mode)

        self.cam = 0
        self.show_display = show_display
        self.cap = None
        self.is_running = False

    def start(self):
        self.cap = cv2.VideoCapture(self.cam)
        self.recorder.start()

        self.is_running = True
        frame_index = 0

        # ⏱️ 1. Restaurando o timer do benchmark
        start_time = time.time()

        was_blinking = False

        # 🛡️ 2. Proteção para pegar o nome do modo (evita erro se for int)
        try:
            current_mode_name = self.inference.mode.name
        except AttributeError:
            current_mode_name = str(self.inference.mode)

        print(f"Watcher iniciado no modo: {current_mode_name}")

        while self.is_running:
            # Primeira leitura (pode ser descartada se precisarmos limpar buffer)
            ret, frame = self.cap.read()
            if not ret:
                break

            now = time.time()
            frame_index += 1

            # Pergunta ao Inference o que fazer
            can_infer = self.inference.can_infer(now)

            # 🧹 Lógica de recuperação de buffer (Anti-Lag)
            # Se acordamos agora (can_infer) e estávamos dormindo (was_blinking)
            if can_infer and was_blinking:
                # Jogamos fora 4 frames velhos do buffer
                for _ in range(4):
                    self.cap.grab()
                # 📸 Lemos o frame ATUALIZADO para processar
                ret, frame = self.cap.read()
                if not ret: break

            # Atualiza estado para o próximo loop
            was_blinking = not can_infer

            if can_infer:
                # 🟢 MODO ATIVO
                t_start = time.perf_counter()
                frame, detections = self.detector.infer_draw(frame)
                t_end = time.perf_counter()

                infer_time_ms = (t_end - t_start) * 1000

                # ✅ Passamos o nome do modo para o CSV
                self.recorder.record(frame_index, infer_time_ms, detections, current_mode_name)
            else:
                # 🔴 MODO PISCADA
                # Registra 0ms, lista vazia e o nome do modo
                self.recorder.record(frame_index, 0, [], current_mode_name)

                # Dorme o tempo que o Inference mandar
                time.sleep(self.inference.get_sleep_time())

            # 📺 Exibição
            if self.show_display:
                cv2.imshow("Watcher feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False

            # ⏱️ 3. Checagem do limite de 30 segundos
            elapsed_time = time.time() - start_time
            if elapsed_time > 30:
                self.is_running = False
                print("Tempo limite de benchmark (30s) excedido.")

        # Finalização
        self.recorder.stop()
        self.recorder.save_csv("data/benchmark.csv")
        self.recorder.save_env_info()
        print(self.recorder.summary())

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()