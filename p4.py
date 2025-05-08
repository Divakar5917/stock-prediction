import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Download AAPL & MSFT data
tickers = ['AAPL', 'MSFT']
data_frames = []

for ticker in tickers:
    df = yf.download(ticker, start='2010-01-01', end='2024-12-31')
    df = df[['Close']].rename(columns={'Close': ticker})
    data_frames.append(df)

# Combine and drop NaNs
combined_df = pd.concat(data_frames, axis=1).dropna()

# 2. Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(combined_df)

# 3. Split into train/test
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size - 60:]  # overlap for sequence

# 4. Create sequences
def create_multivariate_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i])
        y.append(data[i])  # full vector: [AAPL, MSFT]
    return np.array(X), np.array(y)

X_train, y_train = create_multivariate_sequences(train_data)
X_test, y_test = create_multivariate_sequences(test_data)

# 5. Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 6. Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 7. Initialize and train model
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# 8. Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_test).cpu().numpy()
    actual = y_test.cpu().numpy()

# 9. Inverse scale predictions
predicted_prices = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(actual)

# 10. Plot results
plt.figure(figsize=(14, 5))
plt.plot(real_prices[:, 0], color='black', label='AAPL Actual')
plt.plot(predicted_prices[:, 0], color='green', label='AAPL Predicted')
plt.plot(real_prices[:, 1], color='blue', label='MSFT Actual')
plt.plot(predicted_prices[:, 1], color='orange', label='MSFT Predicted')
plt.title("AAPL & MSFT Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
