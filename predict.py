import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

class TouristPredictor:
    """Utility class for making predictions with trained LSTM model"""
    
    def __init__(self, model_path='lstm_model.h5'):
        """Initialize predictor with trained model
        
        Args:
            model_path: path to saved LSTM model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first using app.py")
        
        self.model = load_model(model_path)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._initialize_scaler()
        print(f"✓ Loaded model from {model_path}")
    
    def _initialize_scaler(self):
        """Initialize scaler using historical data"""
        data = pd.read_csv("dataset/date_sorted_tourists.csv")
        values = data[['Value']].values
        self.scaler.fit(values)
    
    def predict_single(self, last_12_months):
        """Predict next month's tourists
        
        Args:
            last_12_months: list/array of last 12 months' values
        
        Returns:
            predicted value (original scale)
        """
        if len(last_12_months) != 12:
            raise ValueError("Expected exactly 12 months of data")
        
        # Normalize
        normalized = self.scaler.transform(np.array(last_12_months).reshape(-1, 1))
        sequence = normalized.reshape(1, 12, 1)
        
        # Predict
        prediction = self.model.predict(sequence, verbose=0)
        
        # Inverse transform
        result = self.scaler.inverse_transform(prediction)[0, 0]
        return result
    
    def predict_multiple(self, last_12_months, periods=12):
        """Predict next N months
        
        Args:
            last_12_months: list/array of last 12 months' values
            periods: number of months to predict
        
        Returns:
            array of predictions
        """
        if len(last_12_months) != 12:
            raise ValueError("Expected exactly 12 months of data")
        
        # Normalize
        normalized = self.scaler.transform(np.array(last_12_months).reshape(-1, 1))
        sequence = normalized.reshape(1, 12, 1)
        
        predictions = []
        for _ in range(periods):
            pred = self.model.predict(sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence with new prediction
            sequence = np.append(sequence[:, 1:, :], [[[pred[0, 0]]]], axis=1)
        
        # Inverse transform all predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
    
    def interactive_predict(self):
        """Interactive prediction mode"""
        print("\n" + "="*60)
        print("INTERACTIVE PREDICTION MODE")
        print("="*60)
        print("\nEnter the last 12 months of tourist data (one value per line)")
        print("Enter 'done' when finished\n")
        
        values = []
        for month in range(1, 13):
            while True:
                try:
                    user_input = input(f"Month {month}: ").strip()
                    if user_input.lower() == 'done':
                        break
                    value = float(user_input)
                    values.append(value)
                    break
                except ValueError:
                    print("Please enter a valid number")
        
        if len(values) == 12:
            print("\nPredicting next month...")
            prediction = self.predict_single(values)
            print(f"\n✓ Predicted tourists for next month: {prediction:,.0f}")
            
            multi_pred = self.predict_multiple(values, periods=6)
            print("\nNext 6 months forecast:")
            for i, pred in enumerate(multi_pred, 1):
                print(f"  Month {i}: {pred:,.0f}")
        else:
            print(f"\n✗ Error: Expected 12 values, got {len(values)}")

def main():
    """Main function for prediction utility"""
    print("\n" + "="*60)
    print("NEPAL TOURISM PREDICTION - INFERENCE MODE")
    print("="*60 + "\n")
    
    # Try to load model
    try:
        predictor = TouristPredictor('lstm_model.h5')
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print("\nPlease run 'python app.py' first to train the model.\n")
        return
    
    # Load latest data for reference
    print("\nLoading historical data...")
    data = pd.read_csv("dataset/date_sorted_tourists.csv")
    last_12 = data['Value'].tail(12).values.tolist()
    
    print(f"✓ Latest 12 months data loaded")
    print(f"  Date range: {data['Date'].iloc[-12]} to {data['Date'].iloc[-1]}")
    
    # Menu
    print("\n" + "-"*60)
    print("OPTIONS:")
    print("-"*60)
    print("1. Predict next month (using latest data)")
    print("2. Predict next 12 months (using latest data)")
    print("3. Custom prediction (enter your own data)")
    print("4. Exit")
    print("-"*60)
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        pred = predictor.predict_single(last_12)
        print(f"\n✓ Predicted tourists for next month: {pred:,.0f}")
    
    elif choice == '2':
        preds = predictor.predict_multiple(last_12, periods=12)
        print("\n✓ 12-Month Forecast:")
        last_date = pd.to_datetime(data['Date'].iloc[-1])
        for i, pred in enumerate(preds, 1):
            future_date = last_date + pd.DateOffset(months=i)
            print(f"  {future_date.strftime('%B %Y'):20s}: {pred:>10,.0f}")
    
    elif choice == '3':
        predictor.interactive_predict()
    
    elif choice == '4':
        print("\n👋 Goodbye!\n")
        return
    
    else:
        print("\n✗ Invalid option\n")

if __name__ == "__main__":
    main()
