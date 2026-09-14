from pydantic import BaseModel


class GesturePredictionData(BaseModel):
    modelId: str = "default"
    rawData: list[list[float]]


class SequencePredictionData(BaseModel):
    gestureModelId: str = "default"
    divisionModelId: str = "default"
