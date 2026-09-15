from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import (
    app,
    division_service,
    gesture_service,
    lifespan,
    storage_service,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_local_models():
    gesture_service.local_models.clear()
    division_service.local_models.clear()
    yield
    gesture_service.local_models.clear()
    division_service.local_models.clear()


@pytest.mark.asyncio
async def test_lifespan():
    mock_rabbitmq = AsyncMock()

    with patch("app.main.rabbitmq", mock_rabbitmq):
        async with lifespan(app):
            mock_rabbitmq.connect.assert_awaited_once()
            assert mock_rabbitmq.declare_queue.await_count == 2


def test_init_gesture_model_success():
    with patch.object(
        storage_service, "load_gesture_model", AsyncMock(return_value=Mock())
    ):
        response = client.post("/models/gesture/test_model")

    assert response.status_code == 200
    assert response.json() == {"message": "Gesture model initialized successfully"}
    assert "test_model" in gesture_service.local_models


def test_init_gesture_model_already_exists():
    gesture_service.local_models["existing_model"] = Mock()

    response = client.post("/models/gesture/existing_model")

    assert response.status_code == 200
    assert response.json() == {"message": "Gesture model already initialized"}


def test_delete_gesture_model_not_found():
    response = client.delete("/models/gesture/missing")

    assert response.status_code == 400
    assert response.json()["detail"] == "Gesture model not found"


def test_init_division_model_failure():
    with patch.object(
        storage_service,
        "load_division_model",
        AsyncMock(side_effect=Exception("Load error")),
    ):
        response = client.post("/models/division/fail_model")

    assert response.status_code == 400
    assert "Failed to initialize division model" in response.json()["detail"]


def test_predict_gesture_success():
    gesture_service.local_models["test_model"] = Mock()

    with patch.object(
        gesture_service,
        "predict",
        AsyncMock(return_value={"predictedLabel": "test", "confidence": 0.95}),
    ):
        response = client.post(
            "/predict/gesture",
            json={
                "modelId": "test_model",
                "rawData": [[1.0] * 18 for _ in range(50)],
            },
        )

    assert response.status_code == 200
    assert response.json()["predictedLabel"] == "test"


def test_predict_gesture_model_not_initialized():
    response = client.post(
        "/predict/gesture",
        json={"modelId": "missing", "rawData": [[1.0] * 18]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "No gesture model initialized"


def test_predict_gesture_empty_data():
    gesture_service.local_models["test_model"] = Mock()

    response = client.post(
        "/predict/gesture", json={"modelId": "test_model", "rawData": []}
    )

    assert response.status_code == 400
    assert "empty 'rawData'" in response.json()["detail"]


def test_predict_gesture_invalid_columns():
    gesture_service.local_models["test_model"] = Mock()

    response = client.post(
        "/predict/gesture",
        json={"modelId": "test_model", "rawData": [[1.0, 2.0, 3.0]]},
    )

    assert response.status_code == 400
    assert "Expected 18 columns" in response.json()["detail"]
