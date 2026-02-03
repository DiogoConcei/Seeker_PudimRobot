import time
import platform
import json
import psutil
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

    def record(self, frame_index: int, infer_time_ms: float, detections: tuple):
        """Registra os dados brutos de cada frame."""
        has_person = any(det[0] == 0 for det in detections)

        self.records.append({
            "frame": frame_index,
            "infer_ms": infer_time_ms,
            "has_person": has_person,
            "num_objects": len(detections),
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
        df.to_csv(path, sep=';',index=False)

    def summary(self) -> dict:
        total_time = self.end_time - self.start_time
        total_frames = len(self.records)
        if total_frames == 0: return {}

        df = self.to_dataframe()

        media_pessoas = df[df["has_person"] == True]["infer_ms"].mean()

        return {
            "total_frames": total_frames,
            "total_time_sec": round(total_time, 3),
            "fps_medio": round(total_frames / total_time, 2),
            "infer_ms_geral": round(df["infer_ms"].mean(), 2),
            "infer_ms_pessoas": round(media_pessoas, 2) if pd.notnull(media_pessoas) else "N/A"
        }