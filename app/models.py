from pydantic import BaseModel


class GesturePredictionData(BaseModel):
    ModelId: str = "default"
    rawData: list[list[float]]


class SequencePredictionData(BaseModel):
    gestureModelId: str = "default"
    divisionModelId: str = "default"
