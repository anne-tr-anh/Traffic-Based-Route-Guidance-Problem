#########################################
# Model Training
# This script creates and trains three models - LSTM, GRU, BiLSTM.
#########################################

import data_processing
import pandas as pd
import os
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Concatenate,
    BatchNormalization,
    Attention,
    GlobalAveragePooling1D,
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def create_lstm_model(seq_length, n_features, n_scats, embedding_dim=16, lstm_units=128, dropout_rate=0.2, use_attention=True):

    x_feature = Input((seq_length, n_features), name="feature_input")
    x_scats = Input((seq_length,), dtype="int32", name="scats_input")
    embed_scats = Embedding(n_scats, embedding_dim)(x_scats)

    x = Concatenate(axis=2)([x_feature, embed_scats])
    x = LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)(x)
    x = LSTM(lstm_units // 2, return_sequences=True, dropout=dropout_rate)(x)
    x = LSTM(lstm_units // 4, return_sequences=True, dropout=dropout_rate)(x)  
    if use_attention:
        x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, name="output")(x)

    return Model([x_feature, x_scats], out)

def create_gru_model(seq_length, n_features, n_scats, embedding_dim=16, gru_units=128, dropout_rate=0.2, use_attention=True):

    x_feature = Input((seq_length, n_features), name="feature_input")
    x_scats = Input((seq_length,), dtype="int32", name="scats_input")
    embed_scats = Embedding(n_scats, embedding_dim)(x_scats)
    
    x = Concatenate(axis=2)([x_feature, embed_scats])
    x = GRU(gru_units, return_sequences=True, dropout=dropout_rate)(x)
    x = GRU(gru_units // 2, return_sequences=True, dropout=dropout_rate)(x)
    x = GRU(gru_units // 4, return_sequences=True, dropout=dropout_rate)(x)
    if use_attention:
        x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, name="output")(x)

    return Model([x_feature, x_scats], out)

def create_bidirectional_lstm_model(seq_length, n_features, n_scats, embedding_dim=16, lstm_units=128, dropout_rate=0.2, use_attention=True):

    x_feature = Input((seq_length, n_features), name="feature_input")
    x_scats = Input((seq_length,), dtype="int32", name="scats_input")
    embed_scats = Embedding(n_scats, embedding_dim)(x_scats)

    x = Concatenate(axis=2)([x_feature, embed_scats])
    x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))(x)
    x = Bidirectional(LSTM(lstm_units // 2, return_sequences=True, dropout=dropout_rate))(x)
    x = Bidirectional(LSTM(lstm_units // 4, return_sequences=True, dropout=dropout_rate))(x)  #
    if use_attention:
        x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, name="output")(x)

    return Model([x_feature, x_scats], out)

Model_List = {
    "lstm": create_lstm_model,
    "bilstm": create_bidirectional_lstm_model,
    "gru": create_gru_model,

}

def sequence_based_split(X, y, metadata_df, val_ratio=0.15, test_ratio=0.15):
    unique_dates = pd.to_datetime(metadata_df["target_date"]).dt.date.unique()
    unique_dates = np.sort(unique_dates)
    val_idx = int(len(unique_dates) * (1 - val_ratio - test_ratio))
    test_idx = int(len(unique_dates) * (1 - test_ratio))
    val_date = unique_dates[val_idx]
    test_date = unique_dates[test_idx]

    metadata_df = metadata_df.copy()
    metadata_df["target_date_obj"] = pd.to_datetime(metadata_df["target_date"]).dt.date
    train_mask = metadata_df["target_date_obj"] < val_date
    val_mask = (metadata_df["target_date_obj"] >= val_date) & (metadata_df["target_date_obj"] < test_date)
    test_mask = metadata_df["target_date_obj"] >= test_date

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    meta_train = metadata_df[train_mask].reset_index(drop=True)
    meta_val = metadata_df[val_mask].reset_index(drop=True)
    meta_test = metadata_df[test_mask].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test

def get_callbacks(model_dir, log_dir, patience_es=10, patience_lr=3):

    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_loss", patience=patience_es, restore_best_weights=True))
    callbacks.append(ModelCheckpoint(model_dir / "best.keras", monitor="val_loss", save_best_only=True))
    callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience_lr, min_lr=1e-6))
    callbacks.append(TensorBoard(log_dir=str(log_dir), histogram_freq=1))

    return callbacks

def train_model(model, X_train_inputs, y_train, X_val_inputs, y_val, epochs=100, batch_size=128, learning_rate=1e-3, clipnorm=1.0, model_name="model", output_dir="models"):

    X_val_inputs=data["X_val_inputs"],
    y_val=data["y_val"],
    base = Path(output_dir)

    model_dir = base / "trained_models" / model_name
    log_dir = base / "logs" / model_name

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model.compile(optimizer=Adam(learning_rate, clipnorm=clipnorm), loss="mse", metrics=["mae"])
    callbacks = get_callbacks(model_dir, log_dir)

    start = time.time()
    history = model.fit(X_train_inputs, y_train, validation_data=(X_val_inputs, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    elapsed = time.time() - start

    model.save(model_dir / "final_model.keras")
    print(f"Training completed in {elapsed:.2f} seconds")
    return history


def create_model(model_type, **kwargs):
    # Create model of specified type
    if model_type not in Model_List:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(Model_List.keys())}")
    return Model_List[model_type](**kwargs)

if __name__ == "__main__":

    data = data_processing.load_processed_data(input_dir="processed_data/")

    # concatenate _train and _test to train first, then split again to test
    X_full = np.concatenate([data["X_train"], data["X_test"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_test"]], axis=0)
    meta_full = pd.concat([data["meta_train"], data["meta_test"]], ignore_index=True)

    # split data
    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = sequence_based_split(X_full, y_full, meta_full)

    # prepare embedding inputs
    feature_cols = data["feature_cols"]
    X_train_inputs = data_processing.prepare_inputs(X_train, feature_cols)
    X_val_inputs = data_processing.prepare_inputs(X_val, feature_cols)
    X_test_inputs = data_processing.prepare_inputs(X_test, feature_cols)

    # update data dictionary
    data.update({
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
        "X_train_inputs": X_train_inputs,
        "X_val_inputs": X_val_inputs,
        "X_test_inputs": X_test_inputs,
    })

    models_dir = "models"
    os.makedirs( models_dir, exist_ok=True)

    models_to_train = [
        {
            "type": "gru",
            "model_params": {
                "gru_units": 128,              
                "dropout_rate": 0.3,
                "embedding_dim": 32,
                "use_attention": True,
            },
            "train_params": {"epochs": 100, "batch_size": 128},
        },
        {
            "type": "lstm",
            "model_params": {
                "lstm_units": 128,
                "dropout_rate": 0.3,
                "embedding_dim": 32,
                "use_attention": True,
            },
            "train_params": {"epochs": 100, "batch_size": 128},
        },
        {
            "type": "bilstm",
            "model_params": {
                "lstm_units": 128,
                "dropout_rate": 0.3,
                "embedding_dim": 32,
                "use_attention": True,
            },
            "train_params": {"epochs": 100, "batch_size": 128},
        },
    ]

    # Train each model
    results = {}
    for config in models_to_train:
        model_type = config["type"]
        model_params = config.get("model_params", {})
        train_params = config.get("train_params", {})
        model_name = model_type
        config["name"] = model_name

        model_params.update({
            "seq_length": data["X_train"].shape[1],
            "n_features": data["n_features"],
            "n_scats": data["n_scats"],
        })

        print(f"\n{'='*50}")
        print(f"Creating {model_type} model...")

        # create model
        if "create_model_func" in config:
            model = config["create_model_func"](**model_params)
        else:
            model = create_model(model_type=model_type, **model_params)

        model.summary()

        # Train model 
        training_results = train_model(
            model=model,
            output_dir=models_dir,
            X_train_inputs=data["X_train_inputs"],
            y_train=data["y_train"],
            X_val_inputs=data["X_val_inputs"],
            y_val=data["y_val"],
            model_name=model_name,
            **train_params,
        )

        results[config["name"]] = {
            "model": model,
            "training_results": training_results,
        }