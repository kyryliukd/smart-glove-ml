import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aio_pika
import requests
import websockets
from fastapi import FastAPI

RABBIT_URL = "amqp://guest:guest@localhost:5672/"
STEP = 56

base_url = "http://localhost:8000"
current_dir = Path.cwd()
data_folder = current_dir.parent / "data" / "gesture_merged"
gestures_merged = data_folder / "gestures_merged.json"
test_gesture = data_folder / "excuse-me.json"
test_stream = data_folder / "please_hello_excuse-me.json"

connection: Optional[aio_pika.RobustConnection] = None
channel: Optional[aio_pika.Channel] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global connection, channel

    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust(RABBIT_URL)
    channel = await connection.channel()

    # Ensure queues exist
    await channel.declare_queue("train_tasks_queue", durable=True)
    await channel.declare_queue("train_results_queue", durable=True)

    # Start consumer in the background
    asyncio.create_task(consume_results())

    yield

    # Close
    await connection.close()


app = FastAPI(lifespan=lifespan)


# Send message
@app.post("/send-training-task")
async def test_send_endpoint(model_id: str = "custom") -> dict:
    message_body = {
        "modelId": model_id
    }

    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(message_body).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        ),
        routing_key="train_tasks_queue"
    )

    return {"message": "Message sent to train_tasks_queue"}


# Get training data
@app.get("/api/v1/internal/models/{model_id}/training-data")
async def get_training_data(model_id: str) -> dict:
    if not gestures_merged.exists():
        return {"error": "Data file not found"}

    with gestures_merged.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# Consumer (listens to the queue)
async def consume_results():
    global connection

    channel = await connection.channel()
    queue = await channel.declare_queue("train_results_queue", durable=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                print(f"Received: {message.body.decode()}")
                
                
@app.post("/init_gesture_model")
def init_gesture_model(model_id: str = "custom"):
    response = requests.post(f"{base_url}/models/gesture/{model_id}")
    return response.json()


@app.post("/delete_gesture_model")
def delete_gesture_model(model_id: str = "custom"):
    response = requests.delete(f"{base_url}/models/gesture/{model_id}")
    return response.json()


@app.post("/init_division_model")
def init_division_model(model_id: str = "default"):
    response = requests.post(f"{base_url}/models/division/{model_id}")
    return response.json()


@app.post("/delete_division_model")
def delete_division_model(model_id: str = "default"):
    response = requests.delete(f"{base_url}/models/division/{model_id}")
    return response.json()


@app.post("/predict/gesture")
async def predict_gesture(model_id: str = "custom"):
    with open(test_gesture, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    data = {
        "modelId": model_id,
        "rawData": test_data
    }
    response = requests.post(f"{base_url}/predict/gesture", json=data)
    
    return response.json()

@app.post("/predict/sequence")
async def predict_sequence(model_id: str = "default"):
    with open(test_stream, 'r', encoding='utf-8') as f:
        reconstructed_stream = json.load(f)
    
    model_data = {
        "gestureModelId": model_id,
        "divisionModelId": model_id
    }
    
    gestures = []

    async def send_frames(websocket):
        """Sends frames without waiting for a response"""
        left, right = 0, STEP
        
        while right < len(reconstructed_stream):
            frame = reconstructed_stream[left:right]
            await websocket.send(json.dumps({"data": frame}))
            print(f"Sent frame {right // STEP} (indices {left}-{right})")
            left += STEP
            right += STEP
            await asyncio.sleep(0.03)
            
        frame = reconstructed_stream[left:]
        if not frame:
            frame = [[0 for _ in range(18)]]
        await websocket.send(json.dumps({"data": frame, "status": "end"}))
        
    async def receive_responses(websocket):
        """Receives responses in the background"""
        try:
            while True:
                response = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=20
                )
                print(f"Received response: {response}")
                
                response = json.loads(response)

                if response.get("predictedLabel"):
                    gestures.append(response["predictedLabel"])
                
        except websockets.ConnectionClosed:
            print("Connection closed")
        except asyncio.TimeoutError:
            print("Timeout waiting for a response from the server")

    uri = "ws://localhost:8000/predict/sequence"

    async with websockets.connect(uri) as websocket:
        # Sending config first
        await websocket.send(json.dumps({
            "gestureModelId": model_data["gestureModelId"],
            "divisionModelId": model_data["divisionModelId"],
        }))
        # Waiting for acknowledgement
        ack = await websocket.recv()
        ack = json.loads(ack)
        print(ack)
        
        if ack["status"] != "ready":
            raise KeyboardInterrupt()
        
        send_task = asyncio.create_task(send_frames(websocket))
        receive_task = asyncio.create_task(receive_responses(websocket))

        try:
            await asyncio.gather(send_task, receive_task)
        except KeyboardInterrupt:
            print("Stoping...")
            send_task.cancel()
            receive_task.cancel()
            
    print(f"\n\n Gestures: {gestures}")
    
    return {"gestures": gestures}
