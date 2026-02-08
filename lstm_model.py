import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def load_data():
    """Load the date-sorted tourist data"""
    data = pd.read_csv("dataset/date_sorted_tourists.csv")
    return data

def create_sequences(data, lookback=12):
    """Create sequences for LSTM training
    
    Args:
        data: numpy array of values
        lookback: number of previous months to use as input (default: 12 for annual pattern)
    
    Returns:
        X, y: sequences and targets
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(lookback=12):
    """Build LSTM model architecture
    
    Args:
        lookback: sequence length (input shape)
    
    Returns:
        model: compiled Keras model
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
    """Train the LSTM model
    
    Args:
        model: Keras model
        X_train: training sequences
        y_train: training targets
        epochs: number of epochs
        batch_size: batch size
        validation_split: validation split ratio
    
    Returns:
        history: training history
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance
    
    Args:
        model: trained Keras model
        X_test: test sequences
        y_test: test targets
        scaler: MinMaxScaler for inverse transformation
    
    Returns:
        metrics: dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics, y_pred_rescaled, y_test_rescaled

def plot_results(history, y_test, y_pred, data_dates):
    """Plot training history and predictions
    
    Args:
        history: training history object
        y_test: actual test values
        y_pred: predicted values
        data_dates: dates for x-axis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid()
    
    # Plot training and validation MAE
    axes[0, 1].plot(history.history['mae'], label='Training MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('Model MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid()
    
    # Plot actual vs predicted
    lookback = 12
    test_dates = data_dates[-(len(y_test) + lookback):]
    
    axes[1, 0].plot(test_dates[-len(y_test):], y_test, 'b-', label='Actual', linewidth=2)
    axes[1, 0].plot(test_dates[-len(y_pred):], y_pred, 'r--', label='Predicted', linewidth=2)
    axes[1, 0].set_title('Actual vs Predicted Tourist Numbers')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Number of Tourists')
    axes[1, 0].legend()
    axes[1, 0].grid()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot prediction errors
    errors = y_test.flatten() - y_pred.flatten()
    axes[1, 1].plot(errors, 'g-', linewidth=1)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].fill_between(range(len(errors)), errors, alpha=0.3)
    axes[1, 1].set_title('Prediction Errors')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Error (Actual - Predicted)')
    axes[1, 1].grid()
    
    plt.tight_layout()
    plt.savefig('data visualization/lstm_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Results saved to 'data visualization/lstm_results.png'")

def forecast_future(model, data, scaler, periods=12):
    """Forecast future values
    
    Args:
        model: trained LSTM model
        data: all training data (scaled)
        scaler: MinMaxScaler for inverse transformation
        periods: number of periods to forecast
    
    Returns:
        forecasts: array of forecasted values
    """
    lookback = 12
    last_sequence = data[-lookback:].reshape(1, lookback, 1)
    forecasts = []
    
    for _ in range(periods):
        next_pred = model.predict(last_sequence, verbose=0)
        forecasts.append(next_pred[0, 0])
        
        # Update sequence with new prediction
        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred[0, 0]]]], axis=1)
    
    # Inverse transform forecasts
    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
    return forecasts

def main():
    """Main function to run LSTM model"""
    print("=" * 60)
    print("LSTM Model for Nepal Tourism Prediction")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data = load_data()
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    # Extract values and dates
    values = data[['Value']].values
    dates = pd.to_datetime(data['Date'])
    
    # Normalize data
    print("\n2. Normalizing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)
    
    # Create sequences
    print("\n3. Creating sequences...")
    lookback = 12  # Using 12-month lookback window
    X, y = create_sequences(scaled_data, lookback)
    print(f"   Sequences created: X shape={X.shape}, y shape={y.shape}")
    
    # Split data
    print("\n4. Splitting data (80% train, 20% test)...")
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build model
    print("\n5. Building LSTM model...")
    model = build_lstm_model(lookback)
    print(model.summary())
    
    # Train model
    print("\n6. Training model...")
    history = train_model(model, X_train, y_train, epochs=50, batch_size=32)
    
    # Evaluate model
    print("\n7. Evaluating model...")
    metrics, y_pred, y_test_rescaled = evaluate_model(model, X_test, y_test, scaler)
    print(f"   RMSE: {metrics['RMSE']:.2f}")
    print(f"   MAE: {metrics['MAE']:.2f}")
    print(f"   R² Score: {metrics['R2']:.4f}")
    
    # Plot results
    print("\n8. Plotting results...")
    plot_results(history, y_test_rescaled, y_pred, dates)
    
    # Save model
    print("\n9. Saving model...")
    model.save('lstm_model.h5')
    print("   Model saved as 'lstm_model.h5'")
    
    # Forecast future values
    print("\n10. Forecasting next 12 months...")
    future_forecast = forecast_future(model, scaled_data, scaler, periods=12)
    
    # Create forecast dates
    last_date = dates.iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    
    print("\n" + "=" * 60)
    print("FORECAST FOR NEXT 12 MONTHS")
    print("=" * 60)
    for date, forecast in zip(future_dates, future_forecast):
        print(f"{date.strftime('%Y-%m')}: {forecast[0]:,.0f} tourists")
    
    print("\n" + "=" * 60)
    print("Model training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
