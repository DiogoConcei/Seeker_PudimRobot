import cv2
from threading import Thread


class CameraThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        # Inicia a thread como 'daemon' (morre junto com o programa principal)
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                # Mantém sempre o frame mais novo na memória
                self.ret, self.frame = ret, frame
            else:
                self.stopped = True

    def read(self):
        # Retorna o último frame instantaneamente
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.cap.release()