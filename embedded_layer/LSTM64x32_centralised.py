from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, Embedding, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import random
import Helper_functions

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

'''
LSTM64x32 centralised validation screening on final cohort - testing with embedding layer for householdid
'''

plots_dir = Path("train_val_curve")
plots_dir.mkdir(exist_ok=True) 


data_path = Path("../03_feature_engineering/final_locked_100_normalised.parquet")

max_min_path = Path("../03_feature_engineering/global_weather_scaler.csv")

local_kwh_scaling = Path("../03_feature_engineering/local_kwh_scaler.csv")


HORIZON = 6
WINDOW_SIZE = 24
TARGET_COL = "kwh"


#model input features
feature_cols = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend", "temperature", "humidity"
]

# Set your embedding size (5 is usually plenty for 100-300 houses)
EMBEDDING_DIM = 5 

def build_nn(time_series_shape, num_houses):
    # --- DOOR 1: Time Series Input ---
    ts_input = Input(shape=time_series_shape, name="time_series_input")
    x = LSTM(64, return_sequences=True)(ts_input)
    x = Dropout(0.2)(x)
    ts_features = LSTM(32)(x)
    ts_features = Dropout(0.2)(ts_features)

    # --- DOOR 2: House ID Input ---
    id_input = Input(shape=(1,), name="house_id_input")
    # input_dim must be exactly the number of unique houses
    house_embedding = Embedding(input_dim=num_houses, output_dim=EMBEDDING_DIM)(id_input)
    house_features = Flatten()(house_embedding)

    # --- THE MERGE ---
    merged = Concatenate()([ts_features, house_features])
    output = Dense(HORIZON, activation="linear")(merged)

    # --- WRAP IN MODEL ---
    model = Model(inputs=[ts_input, id_input], outputs=output)
    
    model.summary() 
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model

def train_model(X_train_ts, X_train_ids, y_train, X_val_ts, X_val_ids, y_val, num_houses):
    model = build_nn(X_train_ts.shape[1:], num_houses)

    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    # Notice we pass a LIST of inputs for both x and validation_data
    history = model.fit(
        x=[X_train_ts, X_train_ids], # both need to have the same number of samples ie rows, but can have different shapes for the other dimensions
        y=y_train,
        validation_data=([X_val_ts, X_val_ids], y_val), 
        epochs=500,
        batch_size=256,
        verbose=1,
        callbacks=[es]
    )
    return model, history




df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)   #load global weather scalers and local kwh scalers
#house_ids = sorted(df["LCLid"].unique())[:15]
house_ids = sorted(df["LCLid"].unique())
#results = []

x_train = []
x_train_ids = []
y_train = []
x_val = []
x_val_ids = []
y_val = []

val_data = {}
    
#print(x_val)

#convert training validation and test data into suitable format for lstm then concatenate for all households

#print(df)

for i, house_id in enumerate(house_ids):
    print(f"Processing {i+1}/{len(house_ids)}: {house_id}")

    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
    train_df, val_df, test_df = Helper_functions.get_house_split(df, house_id, feature_cols)

    #get x and y for train and val sets, using helper function to create windows and targets for each house, with specified window size and horizon
    house_x_train, house_y_train = Helper_functions.make_xy(train_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon = HORIZON)
    house_x_val, house_y_val = Helper_functions.make_xy(val_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon = HORIZON)
    

    if len(house_x_train) == 0 or len(house_x_val) == 0:
        print(f"Skipping {house_id}: insufficient samples after windowing")
        continue

    x_train_ids.append(np.full((house_x_train.shape[0], 1), i))  # create an array of shape (n_samples, 1) filled with the house index
    x_train.append(house_x_train)
    y_train.append(house_y_train)

    x_val_ids.append(np.full((house_x_val.shape[0], 1), i))  # create an array of shape (n_samples, 1) filled with the house index
    x_val.append(house_x_val)
    y_val.append(house_y_val)

    #store x_val, y_val and min max kwh values in dictionary sorted by house_id

    val_data[house_id] = {
        "x_val_ts": house_x_val,                                # Renamed to ts
        "x_val_ids": np.full((house_x_val.shape[0], 1), i),     
        "y_val": house_y_val,
        "kwh_min": kwh_min,
        "kwh_max": kwh_max
    }
print("\nConcatenating centralized datasets...")
x_train_global = np.concatenate(x_train, axis=0)
y_train_global = np.concatenate(y_train, axis=0)
x_val_global = np.concatenate(x_val, axis=0)
y_val_global = np.concatenate(y_val, axis=0)
x_train_ids_global = np.concatenate(x_train_ids, axis=0)
x_val_ids_global = np.concatenate(x_val_ids, axis=0)


print("Global X_train shape:", x_train_global.shape)
print("Global Y_train shape:", y_train_global.shape)
print("Global X_val shape:", x_val_global.shape)
print("Global Y_val shape:", y_val_global.shape)

print("Number of houses in val_data:", len(val_data))



#begin training, set seed for reproducability
tf.keras.backend.clear_session()
tf.random.set_seed(69)
np.random.seed(69)
random.seed(69)

# Pass both the TS and ID arrays, plus the total number of houses
NUM_HOUSES = len(house_ids)
model, history = train_model(
    X_train_ts=x_train_global, 
    X_train_ids=x_train_ids_global, 
    y_train=y_train_global, 
    X_val_ts=x_val_global, 
    X_val_ids=x_val_ids_global, 
    y_val=y_val_global,
    num_houses=NUM_HOUSES
)


#plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')

#vertical line at best epoch
best_epoch_idx = int(np.argmin(history.history["val_loss"]))
best_epoch = best_epoch_idx + 1
plt.axvline(x=best_epoch_idx, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')

plt.title('Embedded LSTM64x32: Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, alpha=0.3)

#save plot
plt.savefig(plots_dir / "embedding_layers.png", dpi=200)
plt.close()


#once training finished and train, val curve has been plotted, perform check for household validation sets

print("Evaluating model performance per household")

results = []

for i, house_id in enumerate(val_data.keys(), start = 1):
    print(f"Evaluatiing {i}/{len(val_data)}: {house_id}")

    pred_scaled = model.predict(
        [val_data[house_id]["x_val_ts"], val_data[house_id]["x_val_ids"]], 
        verbose=0
    )   #run inference

    y_scaled = val_data[house_id]["y_val"]

    kwh_min = float(val_data[house_id]["kwh_min"])
    kwh_max = float(val_data[house_id]["kwh_max"])


    metrics, y_raw, pred_raw = Helper_functions.evaluate_predictions_multistep(
        y_scaled=y_scaled,
        pred_scaled=pred_scaled,
        min_val=kwh_min,
        max_val=kwh_max
    )


    results.append({
        "house_id": house_id,
        "rmse_t+1": metrics["rmse_t+1"],
        "rmse_t+2": metrics["rmse_t+2"],
        "rmse_t+3": metrics["rmse_t+3"],
        "rmse_t+4": metrics["rmse_t+4"],
        "rmse_t+5": metrics["rmse_t+5"],
        "rmse_t+6": metrics["rmse_t+6"],
        "mae_t+1": metrics["mae_t+1"],
        "mae_t+2": metrics["mae_t+2"],
        "mae_t+3": metrics["mae_t+3"],
        "mae_t+4": metrics["mae_t+4"],
        "mae_t+5": metrics["mae_t+5"],
        "mae_t+6": metrics["mae_t+6"],
        "mean_rmse_across_horizons": metrics["mean_rmse_across_horizons"],
        "mean_mae_across_horizons": metrics["mean_mae_across_horizons"]
    })

results_df = pd.DataFrame(results)

print("Mean RMSE across horizons:", results_df["mean_rmse_across_horizons"].mean())
print("Median RMSE across horizons:", results_df["mean_rmse_across_horizons"].median())
print("Std RMSE across horizons:", results_df["mean_rmse_across_horizons"].std())
print("Mean MAE across horizons:", results_df["mean_mae_across_horizons"].mean())
print("Mean RMSE t+1:", results_df["rmse_t+1"].mean())
print("Mean RMSE t+2:", results_df["rmse_t+2"].mean())
print("Mean RMSE t+3:", results_df["rmse_t+3"].mean())
print("Mean RMSE t+4:", results_df["rmse_t+4"].mean())
print("Mean RMSE t+5:", results_df["rmse_t+5"].mean())
print("Mean RMSE t+6:", results_df["rmse_t+6"].mean())
print("Houses evaluated:", results_df["house_id"].nunique())

results_df.to_csv("embedded_layer_per_house_validation_eval.csv", index=False)



best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
epochs_run = len(history.history["loss"])
train_loss_at_best_val = float(history.history["loss"][best_epoch - 1])
best_val_loss = float(history.history["val_loss"][best_epoch - 1])
final_train_loss = float(history.history["loss"][-1])
final_val_loss = float(history.history["val_loss"][-1])
generalisation_gap = best_val_loss - train_loss_at_best_val

#summary statistics
summary_df = pd.DataFrame([{
    "model": "embedded_lstm64x32",
    "mean_rmse_across_horizons": results_df["mean_rmse_across_horizons"].mean(),
    "median_rmse_across_horizons": results_df["mean_rmse_across_horizons"].median(),
    "mean_mae_across_horizons": results_df["mean_mae_across_horizons"].mean(),
    "median_mae_across_horizons": results_df["mean_mae_across_horizons"].median(),

    "mean_rmse_t+1": results_df["rmse_t+1"].mean(),
    "mean_rmse_t+2": results_df["rmse_t+2"].mean(),
    "mean_rmse_t+3": results_df["rmse_t+3"].mean(),
    "mean_rmse_t+4": results_df["rmse_t+4"].mean(),
    "mean_rmse_t+5": results_df["rmse_t+5"].mean(),
    "mean_rmse_t+6": results_df["rmse_t+6"].mean(),

    "mean_mae_t+1": results_df["mae_t+1"].mean(),
    "mean_mae_t+2": results_df["mae_t+2"].mean(),
    "mean_mae_t+3": results_df["mae_t+3"].mean(),
    "mean_mae_t+4": results_df["mae_t+4"].mean(),
    "mean_mae_t+5": results_df["mae_t+5"].mean(),
    "mean_mae_t+6": results_df["mae_t+6"].mean(),

    "best_epoch": best_epoch,
    "epochs_run": epochs_run,
    "train_loss_at_best_val": train_loss_at_best_val,
    "best_val_loss": best_val_loss,
    "final_train_loss": final_train_loss,
    "final_val_loss": final_val_loss,
    "generalisation_gap": generalisation_gap
}])

summary_df.to_csv("embedded_layer_summary.csv", index=False)
print(summary_df)

model.save("embedded_layer.keras")




