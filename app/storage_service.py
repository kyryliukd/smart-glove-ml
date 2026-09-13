import asyncio
import json
import os
import tempfile

import joblib
import tensorflow as tf
from minio import Minio

from .division_service import DivisionService
from .gesture_service import GestureService


class MinioStorage:
    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        bucket_name: str,
    ):
        self.client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
        )
        self.bucket_name = bucket_name

    async def save_gesture_model(self, model_id: str, model: GestureService.Model):
        def _save_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                encoder_path = os.path.join(tmpdir, "encoder.pkl")

                tf.keras.models.save_model(model.model, model_path)
                joblib.dump(model.scaler, scaler_path)
                joblib.dump(model.encoder, encoder_path)

                self.client.fput_object(
                    self.bucket_name, f"gesture/{model_id}/model.keras", model_path
                )
                self.client.fput_object(
                    self.bucket_name, f"gesture/{model_id}/scaler.pkl", scaler_path
                )
                self.client.fput_object(
                    self.bucket_name, f"gesture/{model_id}/encoder.pkl", encoder_path
                )

        await asyncio.to_thread(_save_sync)
        print(f"Model {model_id} saved to MinIO")

    async def load_gesture_model(self, model_id: str) -> GestureService.Model:
        def _load_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                encoder_path = os.path.join(tmpdir, "encoder.pkl")

                self.client.fget_object(
                    self.bucket_name, f"gesture/{model_id}/model.keras", model_path
                )
                self.client.fget_object(
                    self.bucket_name, f"gesture/{model_id}/scaler.pkl", scaler_path
                )
                self.client.fget_object(
                    self.bucket_name, f"gesture/{model_id}/encoder.pkl", encoder_path
                )

                keras_model = tf.keras.models.load_model(model_path)
                scaler = joblib.load(scaler_path)
                encoder = joblib.load(encoder_path)

                return GestureService.Model(
                    model=keras_model, scaler=scaler, encoder=encoder
                )

        return await asyncio.to_thread(_load_sync)

    async def save_division_model(self, model_id: str, model: DivisionService.Model):
        def _save_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                has_start_model_path = os.path.join(tmpdir, "has_start_model.keras")
                has_end_model_path = os.path.join(tmpdir, "has_end_model.keras")
                start_norm_model_path = os.path.join(tmpdir, "start_norm_model.keras")
                end_norm_model_path = os.path.join(tmpdir, "end_norm_model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                thresholds_path = os.path.join(tmpdir, "thresholds.json")

                tf.keras.models.save_model(model.has_start_model, has_start_model_path)
                tf.keras.models.save_model(model.has_end_model, has_end_model_path)
                tf.keras.models.save_model(model.start_norm_model, start_norm_model_path)
                tf.keras.models.save_model(model.end_norm_model, end_norm_model_path)
                joblib.dump(model.scaler, scaler_path)
                
                with open(thresholds_path, "w") as f:
                    json.dump(model.thresholds, f, indent=4)

                self.client.fput_object(
                    self.bucket_name, 
                    f"division/{model_id}/has_start_model.keras", has_start_model_path
                )
                self.client.fput_object(
                    self.bucket_name, 
                    f"division/{model_id}/has_end_model.keras", has_end_model_path
                )
                self.client.fput_object(
                    self.bucket_name, 
                    f"division/{model_id}/start_norm_model.keras", start_norm_model_path
                )
                self.client.fput_object(
                    self.bucket_name, 
                    f"division/{model_id}/end_norm_model.keras", end_norm_model_path
                )
                self.client.fput_object(
                    self.bucket_name, f"division/{model_id}/scaler.pkl", scaler_path
                )
                self.client.fput_object(
                    self.bucket_name, f"division/{model_id}/thresholds.json", thresholds_path
                )

        await asyncio.to_thread(_save_sync)
        print(f"Model {model_id} saved to MinIO")

    async def load_division_model(self, model_id: str) -> DivisionService.Model:
        def _load_sync():
            with tempfile.TemporaryDirectory() as tmpdir:
                has_start_model_path = os.path.join(tmpdir, "has_start_model.keras")
                has_end_model_path = os.path.join(tmpdir, "has_end_model.keras")
                start_norm_model_path = os.path.join(tmpdir, "start_norm_model.keras")
                end_norm_model_path = os.path.join(tmpdir, "end_norm_model.keras")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                thresholds_path = os.path.join(tmpdir, "thresholds.json")
                
                self.client.fget_object(
                    self.bucket_name, 
                    f"division/{model_id}/has_start_model.keras", has_start_model_path
                )
                self.client.fget_object(
                    self.bucket_name, 
                    f"division/{model_id}/has_end_model.keras", has_end_model_path
                )
                self.client.fget_object(
                    self.bucket_name, 
                    f"division/{model_id}/start_norm_model.keras", start_norm_model_path
                )
                self.client.fget_object(
                    self.bucket_name, 
                    f"division/{model_id}/end_norm_model.keras", end_norm_model_path
                )
                self.client.fget_object(
                    self.bucket_name, f"division/{model_id}/scaler.pkl", scaler_path
                )
                self.client.fget_object(
                    self.bucket_name, f"division/{model_id}/thresholds.json", thresholds_path
                )
                
                has_start_model = tf.keras.models.load_model(has_start_model_path)
                has_end_model = tf.keras.models.load_model(has_end_model_path)
                start_norm_model = tf.keras.models.load_model(start_norm_model_path)
                end_norm_model = tf.keras.models.load_model(end_norm_model_path)
                scaler = joblib.load(scaler_path)
                
                with open(thresholds_path) as f:
                    thresholds = json.load(f)

                return DivisionService.Model(
                    has_start_model=has_start_model,
                    has_end_model=has_end_model,
                    start_norm_model=start_norm_model,
                    end_norm_model=end_norm_model,
                    scaler=scaler,
                    thresholds=thresholds
                )

        return await asyncio.to_thread(_load_sync)
