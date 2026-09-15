# Smart Glove ML

[![Dev CI Pipeline](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/dev-ci.yml/badge.svg)](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/dev-ci.yml)

[![Prod CI Pipeline](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/prod-ci.yml/badge.svg)](https://github.com/DmytroKyryliuk2023/smart-glove-ml/actions/workflows/prod-ci.yml)

Machine-learning service for recognizing gestures from Smart Glove sensor data. The service exposes a FastAPI application, stores trained models in MinIO, receives training jobs through RabbitMQ, and communicates with the Smart Glove backend to obtain training data.

## Contents

- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Repository Layout](#repository-layout)
- [Configuration](#configuration)
- [Local Development](#local-development)
- [Docker Compose](#docker-compose)
- [API](#api)
- [Model Processing](#model-processing)
- [Training Flow](#training-flow)
- [Testing and Linting](#testing-and-linting)
- [Git-Ignored Files](#git-ignored-files)

## Architecture

The application has two model pipelines:

1. **Gesture pipeline** classifies a normalized gesture sequence into a label.
2. **Division pipeline** detects gesture start and end positions in a stream.

`GestureDetectionService` combines both pipelines for the WebSocket stream: `DivisionService` finds candidate boundaries, then `GestureService` recognizes the extracted gesture.

```text
Smart Glove Backend
        |
        | training data and model requests
        v
FastAPI application (app/main.py)
        |-- GestureService
        |-- DivisionService
        |-- GestureDetectionService
        |-- TrainingService ---- HTTP ----> Smart Glove Backend
        |-- RabbitMQService --------------> RabbitMQ
        `-- MinioStorage -----------------> MinIO
```

At startup, the application connects to RabbitMQ and starts consuming `train_tasks_queue`. Trained models are uploaded to the `models` MinIO bucket and can later be loaded into the in-memory model registries.

## Technology Stack

- Python 3.12+
- FastAPI and Uvicorn
- TensorFlow / Keras
- NumPy and pandas
- scikit-learn
- RabbitMQ with `aio-pika`
- MinIO with the MinIO Python client
- Pytest and pytest-asyncio
- Ruff
- Docker and Docker Compose

## Repository Layout

```text
smart-glove-ml/
|-- app/
|   |-- main.py                     # FastAPI app and HTTP/WebSocket routes
|   |-- config.py                   # Environment-backed settings
|   |-- models.py                   # Request models
|   |-- gesture_service.py          # Gesture training, resampling, prediction
|   |-- division_service.py         # Stream boundary prediction
|   |-- gesture_detection_service.py# Stream processing and recognition
|   |-- training_service.py         # Training-job orchestration
|   |-- rabbitmq_service.py         # RabbitMQ connection and messaging
|   `-- storage_service.py          # Gesture/division model storage in MinIO
|-- data/
|   |-- gesture/                    # Raw gesture recordings
|   |-- division/                   # Stream boundary training data
|   `-- gesture_merged/             # Prepared gesture datasets
|-- models/
|   |-- gesture/                    # Local gesture model artifacts
|   `-- division/                   # Local division model artifacts
|-- minio-uploader/                 # Image/script for initial MinIO upload
|-- notebooks/                      # EDA, training, prediction, and data prep
|-- tests/                          # Unit tests and shared pytest fixtures
|-- test_server/                    # Small local backend stub
|-- Dockerfile
|-- docker-compose.yml              # RabbitMQ, MinIO, uploader, and ML service
|-- requirements.txt                # Runtime dependencies
|-- requirements.dev.txt            # Development/test dependencies
|-- requirements.eda.txt            # EDA/notebook dependencies
|-- pytest.ini
`-- pyproject.toml                  # Ruff configuration
```

## Configuration

Create a root `.env` file for local execution. `.env` is ignored by Git and must not be committed.

```env
RABBITMQ_URL=amqp://user:password@localhost:5672/
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=models
SERVER_ENDPOINT=http://localhost:8080

# Optional settings; these values are the application defaults.
SEQUENCE_LENGTH=50
NUM_FEATURES=18
WINDOW_SIZE=223
CLOSE_POINTS_THRESHOLD=30
MIN_GESTURE_LENGTH=100
```

`app/config.py` loads these values through Pydantic Settings. The required values are `RABBITMQ_URL`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, and `SERVER_ENDPOINT`. `MINIO_BUCKET_NAME` defaults to `models`.

## Local Development

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install runtime and development dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

Start RabbitMQ and MinIO separately, or use Docker Compose. Then start the application:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The service is available at `http://localhost:8000`.

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker Compose

The root Compose file starts:

- RabbitMQ on ports `5672` and `15672`
- MinIO API on port `9000`
- MinIO Console on port `9001`
- the ML service on port `8000`
- the one-shot MinIO uploader for bundled model artifacts

Configure the credentials used by Compose, then run:

```bash
docker compose up -d --build
```

Stop the stack:

```bash
docker compose down
```

The Compose service passes the ML container these internal values: `rabbitmq:5672`, `minio:9000`, and `http://host.docker.internal:8080` for the backend endpoint.

## API

### Load and delete gesture models

```text
POST   /models/gesture/{model_id}
DELETE /models/gesture/{model_id}
```

Loading downloads the model, scaler, and encoder from MinIO into the local gesture model registry. Deleting removes the model from that registry.

### Load and delete division models

```text
POST   /models/division/{model_id}
DELETE /models/division/{model_id}
```

Division models contain the start/end classifiers, start/end normalization models, scaler, and detection thresholds.

### Predict one gesture

```text
POST /predict/gesture
```

The requested gesture model must already be loaded. Each sensor row must have 18 features.

Request:

```json
{
  "modelId": "default",
  "rawData": [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
  ]
}
```

Response:

```json
{
  "predictedLabel": "hello",
  "confidence": 0.95
}
```

### Predict from a stream

```text
WebSocket /predict/sequence
```

The client sends an initial configuration:

```json
{
  "gestureModelId": "default",
  "divisionModelId": "default"
}
```

After both models are loaded, the server responds with:

```json
{
  "status": "ready",
  "message": "All the models are available"
}
```

The client then sends stream chunks:

```json
{
  "status": "streaming",
  "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]]
}
```

The final chunk uses `"status": "end"`. Recognition responses contain the predicted gesture, its confidence, and detected bounds. Invalid configuration or unavailable models result in an error response and a WebSocket close code.

## Training Flow

Training is initiated by publishing a JSON message to RabbitMQ queue `train_tasks_queue`:

```json
{
  "modelId": "default"
}
```

`TrainingService` then:

1. Requests training data from `{SERVER_ENDPOINT}/api/v1/internal/models/{model_id}/training-data`.
2. Calls `GestureService.train`.
3. Saves the resulting Keras model, scaler, and encoder to MinIO.
4. Publishes a result to `train_results_queue`.

Successful result:

```json
{
  "modelId": "default",
  "status": "SUCCESS",
  "errorMessage": null
}
```

## Model Processing

Gesture sequences are normalized to 50 time steps before training and prediction:

- shorter sequences are linearly interpolated;
- longer sequences are uniformly downsampled;
- sequences already containing 50 rows are kept unchanged.

The default sensor width is 18 features. Division inference uses a sliding window of 223 rows and converts normalized start/end positions into absolute stream indexes.

## Testing and Linting

The unit tests are located directly in `tests/`.

Run all tests:

```bash
pytest tests -v
```

Run tests with terminal coverage:

```bash
pytest tests --cov=app --cov-report=term-missing
```

Generate an HTML coverage report locally:

```bash
pytest tests --cov=app --cov-report=html
```

Run Ruff:

```bash
ruff check app tests
```

The GitHub Actions workflows install `requirements.dev.txt`, run Ruff, and execute the tests. Integration deployment steps are enabled by the workflow configuration when requested.

## Git-Ignored Files

The repository intentionally does not track local environment and generated files, including:

- `.env` and virtual environments such as `.venv/`;
- Python caches, `.pytest_cache/`, coverage data, and `htmlcov/`;
- notebook checkpoints and other IPython local state;
- build, packaging, and tool caches.

These files may appear in a local checkout but should not be added to commits. The model files under `models/` and the bundled uploader files under `minio-uploader/files/` are repository artifacts and are separate from the runtime-generated caches ignored above.

## License

This project is licensed under the MIT License.

## Author

Developed as part of the Smart Glove project and coursework at Lviv Polytechnic National University.
