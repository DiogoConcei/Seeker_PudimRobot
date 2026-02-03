import cv2

class FrameGate:
    #  threshold padrão. 0.01 a 0.03.
    def __init__(self, threshold: float = 0.015, downscale: int = 32):
        self.threshold = threshold
        self.downscale = downscale
        self.last_frame = None

    def should_process(self, frame) -> bool:
        if frame is None: return False

        # Resize é rápido, mas processar GRAY é mais leve
        small = cv2.resize(frame, (self.downscale, self.downscale))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Se for o primeiro frame, processa para criar referência
        if self.last_frame is None:
            self.last_frame = gray
            return True

        # Calcula a diferença
        diff = cv2.absdiff(self.last_frame, gray)
        score = diff.mean() / 255.0

        self.last_frame = gray # Atualiza a referência

        # Debug: Tire o comentário abaixo para calibrar o threshold
        # print(f"Motion Score: {score:.4f} (Threshold: {self.threshold})")

        return score >= self.threshold