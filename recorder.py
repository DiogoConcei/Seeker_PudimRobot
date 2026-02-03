import time
import pandas as pd

class Recorder:
    def __init__(self):
        self.records = []
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def record(self, frame_index: int, infer_time_ms: float, num_detections: int):
        """Registra uma inferência."""
        self.records.append((
            frame_index,
            infer_time_ms,
            num_detections,
            time.perf_counter()
        ))

    def stop(self):
        self.end_time = time.perf_counter()


    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.records,
            columns=[
                "frame_index",
                "infer_time_ms",
                "num_detections",
                "timestamp"
            ]
        )

    def save_csv(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def summary(self) -> dict:
        total_time = self.end_time - self.start_time
        total_frames = len(self.records)

        if total_frames == 0:
            return {}

        df = self.to_dataframe()

        return {
            "total_frames": total_frames,
            "total_time_sec": round(total_time, 3),
            "fps_medio": round(total_frames / total_time, 2),
            "infer_time_medio_ms": round(df["infer_time_ms"].mean(), 2),
            "infer_time_max_ms": round(df["infer_time_ms"].max(), 2),
            "infer_time_min_ms": round(df["infer_time_ms"].min(), 2),
        }




