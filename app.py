import pandas as pd
import matplotlib.pyplot as plt
import sys
from lstm_model import (
    load_data, 
    create_sequences, 
    build_lstm_model, 
    train_model,
    evaluate_model,
    plot_results,
    forecast_future
)
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def main():
    """Main application for Nepal Tourism Prediction"""
    
    print("\n" + "="*70)
    print("NEPAL TOURISM PREDICTION SYSTEM - LSTM MODEL")
    print("="*70 + "\n")
    
    # Load preprocessed data
    print("1. Loading preprocessed tourist data...")
    data = load_data()
    print(f"   ✓ Loaded {len(data)} months of data")
    print(f"   ✓ Date range: {data['Date'].min()} to {data['Date'].max()}\n")
    
    # Extract values and dates
    values = data[['Value']].values
    dates = pd.to_datetime(data['Date'])
    
    # Normalize data
    print("2. Normalizing data using MinMax scaling...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)
    print("   ✓ Data normalized to range [0, 1]\n")
    
    # Create sequences
    print("3. Creating LSTM sequences (lookback=12 months)...")
    lookback = 12
    X, y = create_sequences(scaled_data, lookback)
    print(f"   ✓ Created {len(X)} sequences\n")
    
    # Split data
    print("4. Splitting data into train/test sets...")
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print(f"   ✓ Training samples: {len(X_train)}")
    print(f"   ✓ Testing samples: {len(X_test)}\n")
    
    # Build model
    print("5. Building LSTM architecture...")
    model = build_lstm_model(lookback)
    print("   ✓ Model built successfully\n")
    
    # Train model
    print("6. Training LSTM model...")
    print("   (This may take a few minutes...)\n")
    history = train_model(model, X_train, y_train, epochs=50, batch_size=32)
    print("   ✓ Training completed\n")
    
    # Evaluate model
    print("7. Evaluating model performance...")
    metrics, y_pred, y_test_rescaled = evaluate_model(model, X_test, y_test, scaler)
    print(f"   ✓ RMSE: {metrics['RMSE']:,.2f} tourists")
    print(f"   ✓ MAE: {metrics['MAE']:,.2f} tourists")
    print(f"   ✓ R² Score: {metrics['R2']:.4f}\n")
    
    # Plot results
    print("8. Generating visualizations...")
    plot_results(history, y_test_rescaled, y_pred, dates)
    
    # Save model
    print("\n9. Saving trained model...")
    model.save('lstm_model.h5')
    print("   ✓ Model saved as 'lstm_model.h5'\n")
    
    # Forecast future
    print("10. Generating forecasts for next 12 months...")
    future_forecast = forecast_future(model, scaled_data, scaler, periods=12)
    last_date = dates.iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    
    print("\n" + "="*70)
    print("FORECAST: EXPECTED TOURIST ARRIVALS (NEXT 12 MONTHS)")
    print("="*70)
    for date, forecast in zip(future_dates, future_forecast):
        print(f"  {date.strftime('%B %Y'):20s} : {forecast[0]:>10,.0f} tourists")
    print("="*70 + "\n")
    
    # Save forecast to CSV
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_Tourists': future_forecast.flatten()
    })
    forecast_df.to_csv('forecast_12months.csv', index=False)
    print("✓ Forecast saved to 'forecast_12months.csv'\n")

if __name__ == "__main__":
    main())