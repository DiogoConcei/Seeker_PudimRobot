import time
import platform
import json
import psutil
import os
import pandas as pd


class Recorder:
    def __init__(self):
        self.records = []
        self.start_time = None
        self.end_time = None

    def _get_hardware_info(self):
        info = {
            "SO": f"{platform.system()} {platform.release()}",
            "Processador": platform.processor(),
            "Núcleos_CPU": psutil.cpu_count(logical=True),
            "RAM_Total_GB": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        }
        return info

    def save_env_info(self, path="data/computer.json"):
        info = self._get_hardware_info()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)

    def start(self):
        self.start_time = time.perf_counter()

    def record(self, frame_index: int, infer_time_ms: float, detections: list, mode_name):
        """Registra os dados brutos de cada frame."""
        has_person = any(det[0] == 0 for det in detections)

        hw_fps = (1000 / infer_time_ms) if infer_time_ms > 0 else 0

        self.records.append({
            "frame": frame_index,
            "infer_ms": infer_time_ms,
            "has_person": has_person,
            "num_objects": len(detections),
            "mode": mode_name,
            "hw_fps": round(hw_fps, 2),
            "timestamp": time.perf_counter()
        })

    def stop(self):
        self.end_time = time.perf_counter()

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)

        hw = self._get_hardware_info()

        df["dispositivo"] = hw["Processador"]
        df["ram_gb"] = hw["RAM_Total_GB"]

        return df

    def save_csv(self, path: str):
        df = self.to_dataframe()

        file_exists = os.path.isfile(path)

        df.to_csv(path, sep=';', index=False, mode='a', header=not file_exists)

    def summary(self) -> dict:
        total_time = self.end_time - self.start_time
        total_frames = len(self.records)
        if total_frames == 0: return {}

        df = self.to_dataframe()

        # 1. Filtra apenas frames onde HOUVE inferência (tempo > 0)
        # Ignora os momentos de 'sleep'
        df_active = df[df["infer_ms"] > 0]

        # Média de tempo apenas quando estava trabalhando
        mean_active_ms = df_active["infer_ms"].mean() if not df_active.empty else 0

        # 2. O Cálculo da "Justiça" (Hardware FPS)
        # Se a média for 100ms, o hardware aguenta 10 FPS (1000 / 100)
        hw_potential_fps = (1000 / mean_active_ms) if mean_active_ms > 0 else 0

        # Cálculo antigo de pessoas
        media_pessoas = df[df["has_person"] == True]["infer_ms"].mean()

        return {
            "total_frames": total_frames,
            "total_time_sec": round(total_time, 3),
            "fps_sist_real": round(total_frames / total_time, 2),  # O que o usuário viu
            "fps_hw_potencial": round(hw_potential_fps, 2),  # O que a placa aguenta (JUSTIÇA ⚖️)
            "infer_ms_media_ativa": round(mean_active_ms, 2),  # Média sem contar os zeros
            "infer_ms_pessoas": round(media_pessoas, 2) if pd.notnull(media_pessoas) else "N/A"
        }