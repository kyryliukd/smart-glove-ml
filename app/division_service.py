import asyncio
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class DivisionService:
    @dataclass
    class Model:
        has_start_model: tf.keras.models.Sequential
        has_end_model: tf.keras.models.Sequential
        start_norm_model: tf.keras.models.Sequential
        end_norm_model: tf.keras.models.Sequential
        scaler: StandardScaler
        thresholds: dict

    def __init__(self, window_size: int, num_features: int):
        self.window_size = window_size
        self.num_features = num_features
        self.local_models: dict[str, DivisionService.Model] = {}

    async def predict(self, model: Model, window_data: list, left: int):
        return await asyncio.to_thread(self._predict_sync, model, window_data, left)

    def _predict_sync(
        self, model: Model, window_data: list, left: int
    ) -> tuple[int, int]:
        window_2d = np.asarray(window_data, dtype=np.float32).reshape(
            -1, self.num_features
        )
        window_scaled = model.scaler.transform(window_2d).reshape(
            1, self.window_size, self.num_features
        )

        has_start_prob = model.has_start_model.predict(window_scaled, verbose=0)
        has_end_prob = model.has_end_model.predict(window_scaled, verbose=0)

        has_start_prob = has_start_prob[0][0]
        has_end_prob = has_end_prob[0][0]

        start, end = None, None

        if has_start_prob >= model.thresholds["has_start"]:
            start_norm = model.start_norm_model.predict(window_scaled, verbose=0)
            local_idx = int(start_norm[0][0] * (self.window_size - 1))
            absolute_idx = left + local_idx
            start = absolute_idx

        if has_end_prob >= model.thresholds["has_end"]:
            end_norm = model.end_norm_model.predict(window_scaled, verbose=0)
            local_idx = int(end_norm[0][0] * (self.window_size - 1))
            absolute_idx = left + local_idx
            end = absolute_idx

        return start, end
