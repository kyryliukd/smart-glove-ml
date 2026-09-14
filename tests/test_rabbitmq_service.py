import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.rabbitmq_service import RabbitMQService


@pytest.mark.asyncio
async def test_connect_retry_logic():
    connection = AsyncMock()
    with patch("app.rabbitmq_service.asyncio.sleep", new_callable=AsyncMock):
        with patch(
            "app.rabbitmq_service.aio_pika.connect_robust",
            new_callable=AsyncMock,
            return_value=connection,
        ) as mock_connect:
            connection.channel = AsyncMock()
            service = RabbitMQService("amqp://test")

            await service.connect()

        mock_connect.assert_called_once_with("amqp://test")
        assert service.connection is connection
        connection.channel.assert_awaited_once()


@pytest.mark.asyncio
async def test_declare_queue():
    """Test queue declaration"""
    service = RabbitMQService("amqp://test")
    service.channel = AsyncMock()
    service.channel.declare_queue = AsyncMock()

    await service.declare_queue("test_queue")

    service.channel.declare_queue.assert_called_once_with("test_queue", durable=True)


@pytest.mark.asyncio
async def test_publish_result():
    """Test publishing result to queue"""
    service = RabbitMQService("amqp://test")
    service.channel = AsyncMock()
    service.channel.default_exchange = AsyncMock()
    service.channel.default_exchange.publish = AsyncMock()

    test_message = {"modelId": "123", "status": "SUCCESS"}

    await service.publish_result("test_queue", test_message)

    call_args = service.channel.default_exchange.publish.call_args
    published_message = call_args[0][0]
    routing_key = call_args[1]["routing_key"]

    assert routing_key == "test_queue"
    assert json.loads(published_message.body.decode()) == test_message


@pytest.mark.asyncio
async def test_close_connection():
    """Test closing connection"""
    service = RabbitMQService("amqp://test")
    mock_connection = AsyncMock()
    service.connection = mock_connection

    await service.close()

    mock_connection.close.assert_called_once()


@pytest.mark.asyncio
async def test_start_consuming():
    service = RabbitMQService("amqp://test")
    service.channel = AsyncMock()

    mock_queue = AsyncMock()

    # Create a proper async context manager that returns an async iterator
    class AsyncContextManager:
        async def __aenter__(self):
            async def async_gen():
                yield AsyncMock()

            return async_gen()

        async def __aexit__(self, *args):
            pass

    # Make iterator a regular Mock that returns the context manager directly
    mock_queue.iterator = Mock(return_value=AsyncContextManager())
    service.channel.declare_queue = AsyncMock(return_value=mock_queue)

    mock_callback = AsyncMock()

    await service.start_consuming("test_queue", mock_callback)
    service.channel.declare_queue.assert_awaited_once_with("test_queue", durable=True)
