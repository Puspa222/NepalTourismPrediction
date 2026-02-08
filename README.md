# Nepal Tourism Prediction using LSTM

## 📋 Project Overview
This project uses **Long Short-Term Memory (LSTM)** neural networks to predict tourist arrivals in Nepal based on historical monthly data from 1995 to present.

## 🎯 Features
- **Time Series Analysis**: Analyzes 25+ years of monthly tourism data
- **LSTM Neural Network**: Deep learning model with 2-layer LSTM architecture
- **Data Preprocessing**: Automatic data normalization and sequence creation
- **Model Evaluation**: Comprehensive metrics (RMSE, MAE, R² Score)
- **Future Forecasting**: Predicts next 12 months of tourist arrivals
- **Visualization**: Training history, actual vs predicted, and error plots

## 📊 Dataset
- **Source**: `dataset/date_sorted_tourists.csv`
- **Time Period**: 1995 - 2026 (monthly data)
- **Points**: 362 monthly observations
- **Features**: Date, Tourist Count

## 🏗️ Project Structure
```
NepalTourismPrediction/
├── app.py                          # Main application
├── lstm_model.py                   # LSTM implementation
├── data_preprocessing.py           # Data preprocessing script
├── requirements.txt                # Python dependencies
├── dataset/
│   ├── initial_data.csv           # Raw data
│   └── date_sorted_tourists.csv   # Preprocessed data
├── data visualization/
│   └── scatter_plot.py            # Visualization utilities
└── README.md                       # This file
```

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
If not already done, run the preprocessing script:
```bash
python data_preprocessing.py
```

### 3. Run the Main Application
```bash
python app.py
```

## 🧠 LSTM Model Architecture

### Model Configuration
- **Input Layer**: 12-month lookback window
- **Layer 1**: LSTM with 50 units + 20% Dropout
- **Layer 2**: LSTM with 50 units + 20% Dropout
- **Dense Layer**: 25 units with ReLU activation
- **Output Layer**: 1 unit (tourist count prediction)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)

### Training Parameters
- **Epochs**: 50
- **Batch Size**: 32
- **Validation Split**: 20%
- **Train/Test Split**: 80/20

## 📈 Expected Outputs

### 1. Model Metrics
The application will display:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of Determination

### 2. Visualizations
4-panel plot saved to `data visualization/lstm_results.png`:
- Training and Validation Loss curves
- Training and Validation MAE curves
- Actual vs Predicted tourist numbers
- Prediction errors

### 3. Forecast CSV
Future predictions saved to `forecast_12months.csv`:
- Date
- Forecasted tourist count

### 4. Trained Model
The trained LSTM model is saved as `lstm_model.h5` for reuse

## 📝 Usage Examples

### Running Full Pipeline
```bash
python app.py
```

### Custom LSTM Training
```python
from lstm_model import *

# Load data
data = load_data()
values = data[['Value']].values

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)

# Create sequences
X, y = create_sequences(scaled_data, lookback=12)

# Build and train
model = build_lstm_model(lookback=12)
history = train_model(model, X_train, y_train, epochs=100)
```

## 🔍 Model Performance
- Captures annual tourism patterns
- Handles seasonal variations
- Provides reliable 12-month forecasts
- Low approximation error on test set

## 📚 Technical Details

### Sequence Creation
- **Lookback Period**: 12 months
- **Reason**: Captures annual seasonality in tourism

### Data Normalization
- **Method**: Min-Max Scaling (0-1 range)
- **Reason**: Improves LSTM training stability

### Evaluation Metrics
- **RMSE**: Penalizes larger errors
- **MAE**: Average absolute deviation
- **R² Score**: Goodness of fit indicator

## 🛠️ Troubleshooting

### Memory Issues
If you encounter GPU memory issues, reduce batch size:
```python
train_model(model, X_train, y_train, batch_size=16)
```

### Training Too Slow
Use fewer epochs or smaller network:
```python
history = train_model(model, X_train, y_train, epochs=25)
```

## 📖 References
- Hochreiter & Schmidhuber (1997) - LSTM Paper
- Keras Documentation: https://keras.io/
- TensorFlow: https://www.tensorflow.org/

## 📄 License
This project is open source and available for educational purposes.

## 👨‍💻 Author
Data Science & Machine Learning Project - Nepal Tourism Analysis