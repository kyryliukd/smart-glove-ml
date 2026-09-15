from unittest.mock import Mock

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.gesture_service import GestureService
from app.models import GesturePredictionData, SequencePredictionData


def test_resample_sequence_interpolation():
    """Test resampling when input is shorter than target"""
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    service = GestureService(sequence_length=50, num_features=2)
    result = service._resample_sequence(df, 50)

    assert len(result) == 50
    assert result.shape[1] == 2
    assert result.iloc[0][0] == 1
    assert result.iloc[-1][0] == 5


def test_resample_sequence_downsample():
    """Test resampling when input is longer than target"""
    df = pd.DataFrame(np.random.rand(100, 5))
    service = GestureService(sequence_length=50, num_features=5)
    result = service._resample_sequence(df, 50)

    assert len(result) == 50
    assert result.shape[1] == 5
    assert all(result.iloc[i] is not None for i in range(len(result)))


def test_resample_sequence_equal_length():
    """Test resampling when input equals target length"""
    df = pd.DataFrame(np.random.rand(50, 5))
    service = GestureService(sequence_length=50, num_features=5)
    result = service._resample_sequence(df, 50)

    assert len(result) == 50
    pd.testing.assert_frame_equal(df.reset_index(drop=True), result)


def test_gesture_prediction_data_validation():
    data = GesturePredictionData(modelId="test_model", rawData=[[1.0, 2.0], [3.0, 4.0]])
    assert data.modelId == "test_model"
    assert len(data.rawData) == 2


def test_sequence_prediction_data_defaults():
    data = SequencePredictionData()

    assert data.gestureModelId == "default"
    assert data.divisionModelId == "default"


def test_model_dataclass():
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array([["a"], ["b"]]))
    mock_model = Mock()

    model_obj = GestureService.Model(model=mock_model, scaler=scaler, encoder=encoder)

    assert model_obj.model == mock_model
    assert model_obj.scaler == scaler
    assert model_obj.encoder == encoder
