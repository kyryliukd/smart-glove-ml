from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.training_service import TrainingService


@pytest.fixture
def training_service():
    return TrainingService(
        rabbitmq=Mock(),
        gesture_service=Mock(),
        storage_service=Mock(),
        server_endpoint="http://server",
    )


@pytest.mark.asyncio
async def test_fetch_training_data(training_service):

    mock_response = Mock()
    mock_response.json = Mock(return_value={"gesture1": []})
    mock_response.raise_for_status = Mock()

    with patch("app.training_service.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await training_service.fetch_training_data("test_id")

        assert result == {"gesture1": []}


@pytest.mark.asyncio
async def test_train_gesture_model_success(training_service, training_data):
    trained_model = Mock()
    training_service.gesture_service.train = AsyncMock(return_value=trained_model)
    training_service.storage_service.save_gesture_model = AsyncMock()

    with patch.object(
        training_service, "fetch_training_data", AsyncMock(return_value=training_data)
    ):
        await training_service.train_gesture_model(
            training_service.gesture_service, "test_id"
        )

    training_service.gesture_service.train.assert_awaited_once_with(training_data)
    training_service.storage_service.save_gesture_model.assert_awaited_once_with(
        "test_id", trained_model
    )


def test_service_dependencies_are_stored(training_service):
    assert training_service.server_endpoint == "http://server"
    assert training_service.rabbitmq is not None
