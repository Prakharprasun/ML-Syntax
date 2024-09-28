import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load stock data (example: AAPL stock data)
data = pd.read_csv('AAPL.csv')  # CSV with columns: Date, Open, High, Low, Close, Volume
data = data[['Close']]  # Focus only on Close prices

# Scale the data to be between 0 and 1 for better performance
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Prepare the data for the GRU
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Create dataset for 60 days window
time_step = 60
X, y = create_dataset(data_scaled, time_step)

# Reshape the data to fit the GRU input (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets (80/20 split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the GRU model
model = Sequential()
model.add(GRU(64, return_sequences=False, input_shape=(time_step, 1)))
model.add(Dense(1))  # Predict next day's closing price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions back to original scale
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(y_test, label='True Stock Price')
plt.plot(y_pred, label='Predicted Stock Price')
plt.title('Stock Price Prediction with GRU (TensorFlow)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load stock data (example: AAPL stock data)
data = pd.read_csv('AAPL.csv')  # CSV with columns: Date, Open, High, Low, Close, Volume
data = data[['Close']]  # Focus only on Close prices

# Scale the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Prepare the data for the GRU
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Create dataset for 60 days window
time_step = 60
X, y = create_dataset(data_scaled, time_step)

# Reshape the data to fit the GRU input (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Convert data to PyTorch tensors
X_train = torch.tensor(X[:int(len(X) * 0.8)], dtype=torch.float32)
y_train = torch.tensor(y[:int(len(y) * 0.8)], dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X[int(len(X) * 0.8):], dtype=torch.float32)
y_test = torch.tensor(y[int(len(y) * 0.8):], dtype=torch.float32).unsqueeze(1)

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)  # Get the output of GRU
        out = self.fc(out[:, -1, :])  # Use the last output for prediction
        return out

# Initialize model, loss function, and optimizer
model = GRUModel(input_size=1, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')

# Make predictions
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.detach().numpy()

# Inverse transform predictions back to original scale
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(y_test, label='True Stock Price')
plt.plot(y_pred, label='Predicted Stock Price')
plt.title('Stock Price Prediction with GRU (PyTorch)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
