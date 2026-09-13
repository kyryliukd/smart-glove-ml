import asyncio
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class GestureService:
    @dataclass
    class Model:
        model: tf.keras.models.Sequential
        scaler: StandardScaler
        encoder: OneHotEncoder

    def __init__(self, sequence_length: int, num_features: int):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.local_models: dict[str, GestureService.Model] = {}
        
    def _resample_sequence(self, df: pd.DataFrame, target_length: int) -> pd.DataFrame:
        """
        Adjusts the number of rows in the DataFrame to target_length.
        If there are fewer rows, interpolation is performed.
        If there are more rows, points are selected uniformly.

        Args:
            df: Pandas DataFrame containing time-series data (one gesture).
            target_length: Desired number of points after resampling.

        Returns:
            Pandas DataFrame with the same structure and target_length rows.
        """
        df = df.reset_index(drop=True)
        current_length = len(df)

        if current_length < target_length:
            new_index = np.linspace(0, current_length - 1, target_length)
            df_resampled = df.reindex(new_index)
            df_resampled = df_resampled.interpolate(method="linear")
            return df_resampled.reset_index(drop=True)

        elif current_length > target_length:
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            df_resampled = df.iloc[indices].reset_index(drop=True)
            return df_resampled

        else:
            return df

    async def predict(self, model: Model, gesture_data: list):
        return await asyncio.to_thread(self._predict_sync, model, gesture_data)

    def _predict_sync(self, model: Model, gesture_data: list):
        df = pd.DataFrame(gesture_data)
        df_resampled = self._resample_sequence(df, self.sequence_length)

        input_data = df_resampled.values.astype(float)
        data_scaled = model.scaler.transform(input_data)
        data_for_model = np.expand_dims(data_scaled, axis=0)

        prediction_probs = model.model.predict(data_for_model, verbose=0)[0]
        
        label_index = np.argmax(prediction_probs)
        predicted_label = model.encoder.inverse_transform(
            np.eye(len(prediction_probs))[label_index].reshape(1, -1)
        )[0][0]
        confidence = float(prediction_probs[label_index])

        return {"predictedLabel": predicted_label, "confidence": float(confidence)}

    async def train(self, training_data: dict) -> Model:
        return await asyncio.to_thread(self._train_sync, training_data)

    def _train_sync(self, gestures: dict) -> Model:
        if not gestures:
            raise Exception("Received empty data for training")

        model = GestureService.Model(model=None, scaler=None, classes=None)
        samples, labels = [], []

        for label, sequences in gestures.items():
            for seq in sequences:
                df = pd.DataFrame(seq)
                if df.shape[1] != self.num_features:
                    print(
                        f"Skipping {label} — incorrect number of columns {df.shape[1]}"
                    )
                    continue

                df_resampled = GestureService._resample_sequence(
                    df, self.sequence_length
                )
                if df_resampled.shape != (self.sequence_length, self.num_features):
                    print(
                        f"Skipping {label} after resampling — received {df_resampled.shape}"
                    )
                    continue

                samples.append(df_resampled.values.astype(float))
                labels.append(label)

        if len(samples) == 0:
            raise Exception("No valid data for training")

        samples, labels = np.array(samples), np.array(labels)

        _, counts = np.unique(labels, return_counts=True)
        if np.any(counts < 2):
            raise Exception("Each class must have at least 2 examples")

        X_train, X_test, labels_train, labels_test = train_test_split(
            samples,
            labels,
            test_size=0.3,
            random_state=42,
            stratify=labels,
        )
        
        model.encoder = OneHotEncoder(sparse_output=False)

        y_train = model.encoder.fit_transform(
            labels_train.reshape(-1, 1)
        )

        y_test = model.encoder.transform(
            labels_test.reshape(-1, 1)
        )

        model.scaler = StandardScaler()

        N_train, T, F = X_train.shape
        N_test = X_test.shape[0]

        X_train_2d = X_train.reshape(-1, F)
        X_test_2d = X_test.reshape(-1, F)

        model.scaler.fit(X_train_2d)

        X_train_scaled = model.scaler.transform(X_train_2d).reshape(
            N_train, T, F
        )

        X_test_scaled = model.scaler.transform(X_test_2d).reshape(
            N_test, T, F
        )

        num_classes = y_train.shape[1]

        model.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(T, F)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])

        model.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.model.fit(
            X_train_scaled,
            y_train,
            epochs=30,
            batch_size=16,
            verbose=1,
        )

        _, train_accuracy = model.model.evaluate(
            X_train_scaled,
            y_train,
            verbose=0
        )

        _, test_accuracy = model.model.evaluate(
            X_test_scaled,
            y_test,
            verbose=0
        )

        print(
            "Custom model evaluation:"
            f"  Train Accuracy: {train_accuracy * 100:.2f}%\n"
            f"  Test Accuracy: {test_accuracy * 100:.2f}%"
        )

        return model
