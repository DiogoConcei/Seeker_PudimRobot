from frame_gate import FrameGate
from enum import Enum

class InferenceMode(Enum):
    CONTINUOUS=1
    BLINK_FIX=2
    TIMELESS_BLINK=3
    HYBRID = 4

class Inference:
    def __init__(self):
        # ----- blink fixo (tempo) -----
        self.blink_interval = 1.5
        self.blink_duration = 1.0
        self.last_blink_time = 0.0
        self.blink_start = 0.0
        self.mode = ''
        self.blinking = False

        self.sleep_time = 0.2

        # ---- Hybrid
        self.gate = FrameGate(threshold=0.015, downscale=32)
        self.cooldown_counter = 0
        self.max_cooldown = 30

        # ----- Timeless
        self.blink_cycle_interval = 10.0
        self.last_blink_cycle = 0.0

        self.infer_delays = [2.0, 1.0, 0.5, 0.0]
        self.delay_index = 0
        self.next_infer_time = 0.0
        self.in_adaptive_blink = False

    def get_sleep_time(self):
        return self.sleep_time

    def can_infer(self, now: float, frame=None, last_detections=None) -> bool:
        if self.mode == InferenceMode.CONTINUOUS:
            return True

        if self.mode == InferenceMode.BLINK_FIX:
            return self._can_infer_blink_fixed(now)

        if self.mode == InferenceMode.TIMELESS_BLINK:
            if frame is None: return True
            return self._can_infer_blink_adaptive(frame)

        if self.mode == InferenceMode.HYBRID:
            return self._can_infer_hybrid(frame, last_detections)



        return True

    def _can_infer_blink_fixed(self, now: float) -> bool:
        if not self.blinking:
            if now - self.last_blink_time >= self.blink_interval:
                self.blinking = True
                self.blink_start = now
        else:
            if now - self.blink_start >= self.blink_duration:
                self.blinking = False
                self.last_blink_time = now

        return not self.blinking

    def _can_infer_blink_adaptive(self, frame) -> bool:
        if frame is None:
            return True

        return self.gate.should_process(frame)

    def _can_infer_hybrid(self, frame, last_detections) -> bool:
        """
        Lógica Sentinela:
        1. Se viu pessoa na última vez -> (Cooldown).
        2. Se tem Cooldown -> Roda contínuo.
        3. Se não tem Cooldown -> Usa o Gate (visão periférica).
        """

        # 1. Verifica se vimos alguém no último frame processado
        found_person = False
        if last_detections:
            # Verifica se alguma detecção tem class_id == 0 (Pessoa)
            found_person = any(int(d[0]) == 0 for d in last_detections)

        if found_person:
            self.cooldown_counter = self.max_cooldown
            return True

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return True

        # 3. Se está calmo, usa o Gate (Visão Periférica) para economizar
        if frame is None: return True

        is_moving = self.gate.should_process(frame)
        return is_moving