from unittest.mock import Mock, patch

import pytest

from app.storage_service import MinioStorage


def make_storage(mock_minio_client):
    minio_patcher = patch("app.storage_service.Minio", return_value=mock_minio_client)
    minio_patcher.start()
    return minio_patcher, MinioStorage("minio:9000", "key", "secret", "test-bucket")


@pytest.mark.asyncio
async def test_save_model(mock_minio_client, sample_model):
    minio_patcher, storage = make_storage(mock_minio_client)
    try:
        with patch("app.storage_service.tf.keras.models.save_model") as save_model:
            with patch("app.storage_service.joblib.dump") as dump:
                await storage.save_gesture_model("test_id", sample_model)

        assert save_model.called
        assert dump.call_count == 2
        assert mock_minio_client.fput_object.call_count == 3
    finally:
        minio_patcher.stop()


@pytest.mark.asyncio
async def test_load_model(mock_minio_client):
    minio_patcher, storage = make_storage(mock_minio_client)
    try:
        with patch(
            "app.storage_service.tf.keras.models.load_model", return_value=Mock()
        ):
            with patch("app.storage_service.joblib.load", side_effect=[Mock(), Mock()]):
                result = await storage.load_gesture_model("test_id")

        assert result is not None
        assert mock_minio_client.fget_object.call_count == 3
    finally:
        minio_patcher.stop()
