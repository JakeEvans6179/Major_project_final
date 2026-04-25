import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CNN_LSTM = pd.read_csv("CNN_LSTM.csv")
CNN_LSTM_Dense = pd.read_csv("CNN_LSTM_Dense.csv")
LSTM64x32 = pd.read_csv("LSTM64x32.csv")
LSTM64_Dense = pd.read_csv("LSTM_Dense.csv")

cnn_lstm_val = CNN_LSTM["mean_rmse_kwh"].to_numpy()
cnn_lstm_dense_val = CNN_LSTM_Dense["mean_rmse_kwh"].to_numpy()
lstm64x32_val = LSTM64x32["mean_rmse_kwh"].to_numpy()[:40]
lstm64_dense_val = LSTM64_Dense["mean_rmse_kwh"].to_numpy()

rounds_per_chunk = 5

x1 = np.arange(1, len(cnn_lstm_val) + 1) * rounds_per_chunk
x2 = np.arange(1, len(cnn_lstm_dense_val) + 1) * rounds_per_chunk
x3 = np.arange(1, len(lstm64x32_val) + 1) * rounds_per_chunk
x4 = np.arange(1, len(lstm64_dense_val) + 1) * rounds_per_chunk

plt.figure(figsize=(11, 4))

plt.plot(x1, cnn_lstm_val, label="CNN-LSTM", linewidth=2, color='blue')
plt.plot(x2, cnn_lstm_dense_val, label="CNN-LSTM-Dense", linewidth=2, color = 'orange')
plt.plot(x4, lstm64_dense_val, label="LSTM64-Dense", linewidth=2, color = 'green')
plt.plot(x3, lstm64x32_val, label="LSTM64_32", linewidth=2, color = 'red')


plt.xlabel("Communication Rounds")
plt.ylabel("Validation RMSE (kWh)")
plt.title("Validation RMSE over Communication Rounds")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xticks(np.arange(0, 201, 5))

plt.savefig("validation_rmse_over_communication_rounds.png", dpi=300, bbox_inches="tight")
plt.show()