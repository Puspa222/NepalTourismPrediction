"""ARIMA and SARIMA modeling for Nepal tourism monthly arrivals

Creates and evaluates ARIMA and SARIMA models.
Saves model summaries and forecast plots to `plots/`.

Usage:
    python3 models/arima_sarima.py

Dependencies:
    numpy, pandas, matplotlib, statsmodels, sklearn

"""
import os
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'dataset', 'initial_data.csv')
PLOTS_DIR = os.path.join(ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_series(path=DATA_PATH):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
    df = df.set_index('date').sort_index()
    ts = df['Value'].astype(float)
    ts.name = 'tourists'
    return ts


def train_test_split(ts, test_periods=12):
    train = ts.iloc[:-test_periods]
    test = ts.iloc[-test_periods:]
    return train, test


def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'rmse': rmse, 'mae': mae, 'mape': mape}


def fit_arima_grid(train, max_p=2, max_d=1, max_q=2):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                order = (p, d, q)
                try:
                    model = ARIMA(train, order=order)
                    res = model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = order
                        best_model = res
                except Exception:
                    continue
    return best_order, best_model


def fit_sarima_grid(train, seasonal_period=12, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1):
    best_aic = np.inf
    best_cfg = None
    best_model = None

    p = range(0, max_p + 1)
    d = range(0, max_d + 1)
    q = range(0, max_q + 1)
    P = range(0, max_P + 1)
    D = range(0, max_D + 1)
    Q = range(0, max_Q + 1)

    for order in itertools.product(p, d, q):
        for seasonal_order in itertools.product(P, D, Q):
            try:
                model = SARIMAX(train, order=order, seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_period), enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_cfg = (order, seasonal_order)
                    best_model = res
            except Exception:
                continue

    return best_cfg, best_model


def plot_forecast(train, test, pred, title, fname):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='train')
    plt.plot(test.index, test, label='test')
    plt.plot(pred.index, pred, label='forecast', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, fname), bbox_inches='tight')
    plt.close()


def save_model_summary(res, path):
    with open(path, 'w') as f:
        f.write(res.summary().as_text())


def main():
    ts = load_series()
    train, test = train_test_split(ts, test_periods=12)

    print('Fitting ARIMA grid (this may take a short while)...')
    arima_order, arima_res = fit_arima_grid(train, max_p=2, max_d=1, max_q=2)
    if arima_res is None:
        print('No ARIMA model could be fit.')
    else:
        print('Best ARIMA order:', arima_order)
        save_model_summary(arima_res, os.path.join(PLOTS_DIR, 'arima_summary.txt'))

        # Forecast
        arima_pred = arima_res.get_forecast(steps=len(test)).predicted_mean
        arima_pred.index = test.index
        arima_eval = evaluate_forecast(test.values, arima_pred.values)
        print('ARIMA evaluation:', arima_eval)
        plot_forecast(train, test, arima_pred, f'ARIMA{arima_order} forecast', 'arima_forecast.png')

    print('Fitting SARIMA grid (seasonal period=12)...')
    sarima_cfg, sarima_res = fit_sarima_grid(train, seasonal_period=12, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1)
    if sarima_res is None:
        print('No SARIMA model could be fit.')
    else:
        print('Best SARIMA config:', sarima_cfg)
        save_model_summary(sarima_res, os.path.join(PLOTS_DIR, 'sarima_summary.txt'))

        # Forecast (dynamic forecasting)
        sarima_forecast = sarima_res.get_forecast(steps=len(test))
        sarima_pred = sarima_forecast.predicted_mean
        sarima_pred.index = test.index
        sarima_eval = evaluate_forecast(test.values, sarima_pred.values)
        print('SARIMA evaluation:', sarima_eval)
        plot_forecast(train, test, sarima_pred, f'SARIMA{sarima_cfg[0]} x {sarima_cfg[1]}12 forecast', 'sarima_forecast.png')

    print('Done. Summaries and plots saved in:', PLOTS_DIR)


if __name__ == '__main__':
    main()
