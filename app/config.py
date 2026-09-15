from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    rabbitmq_url: str

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket_name: str = "models"

    server_endpoint: str

    sequence_length: int = 50
    num_features: int = 18

    window_size: int = 223
    close_points_threshold: int = 30
    min_gesture_length: int = 100
    

settings = Settings()