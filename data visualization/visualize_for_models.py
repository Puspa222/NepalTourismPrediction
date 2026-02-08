# Visualizations for ARIMA and LSTM modeling
# Saves plots into a `plots/` directory.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler

sns.set(style='whitegrid')

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'initial_data.csv')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_series(path=DATA_PATH):
    df = pd.read_csv(path)
    # create a datetime index from Year and Month (abbrev)
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
    df = df.set_index('date').sort_index()
    ts = df['Value'].astype(float)
    ts.name = 'tourists'
    return ts


def plot_series(ts):
    plt.figure(figsize=(14,5))
    ts.plot(title='Monthly tourist arrivals')
    plt.ylabel('Arrivals')
    plt.savefig(os.path.join(PLOTS_DIR, 'ts_plot.png'), bbox_inches='tight')
    plt.close()


def plot_decomposition(ts, period=12):
    # multiplicative may be appropriate if variance grows with level; try additive first
    res = seasonal_decompose(ts, model='additive', period=period, two_sided=True, extrapolate_trend='freq')
    fig = res.plot()
    fig.set_size_inches(12, 9)
    fig.suptitle('Seasonal decomposition (additive)')
    plt.savefig(os.path.join(PLOTS_DIR, 'decomposition.png'), bbox_inches='tight')
    plt.close()


def plot_acf_pacf(ts, lags=48):
    fig, axes = plt.subplots(2, 1, figsize=(12,8))
    plot_acf(ts.dropna(), ax=axes[0], lags=lags)
    plot_pacf(ts.dropna(), ax=axes[1], lags=lags, method='ywm')
    axes[0].set_title('ACF')
    axes[1].set_title('PACF')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'acf_pacf.png'), bbox_inches='tight')
    plt.close()


def adf_test_and_plot(ts):
    print('Running ADF test...')
    result = adfuller(ts.dropna())
    # result: (adf_stat, pvalue, usedlag, nobs, crit_values, icbest)
    out = {
        'adf_stat': result[0],
        'pvalue': result[1],
        'usedlag': result[2],
        'nobs': result[3],
        'crit_values': result[4]
    }
    print(out)

    # plot rolling mean/std
    rol_mean = ts.rolling(window=12).mean()
    rol_std = ts.rolling(window=12).std()
    plt.figure(figsize=(14,5))
    plt.plot(ts, label='Original')
    plt.plot(rol_mean, label='12-mo Rolling Mean')
    plt.plot(rol_std, label='12-mo Rolling Std')
    plt.legend()
    plt.title('Rolling statistics')
    plt.savefig(os.path.join(PLOTS_DIR, 'rolling_stats.png'), bbox_inches='tight')
    plt.close()

    return out


def differencing_and_plot(ts, d=1):
    ts_diff = ts.diff(d)
    plt.figure(figsize=(14,5))
    ts_diff.plot(title=f'{d}-order differenced series')
    plt.ylabel('Differenced arrivals')
    plt.savefig(os.path.join(PLOTS_DIR, f'differenced_{d}.png'), bbox_inches='tight')
    plt.close()
    return ts_diff


def season_boxplot(ts):
    df = ts.reset_index()
    df['month'] = df['date'].dt.month_name().str.slice(stop=3)
    order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    plt.figure(figsize=(12,6))
    sns.boxplot(x='month', y='tourists', data=df, order=order)
    plt.title('Monthly distribution (boxplot)')
    plt.savefig(os.path.join(PLOTS_DIR, 'month_boxplot.png'), bbox_inches='tight')
    plt.close()


def heatmap_month_year(ts):
    df = ts.reset_index()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    pivot = df.pivot(index='month', columns='year', values='tourists')
    plt.figure(figsize=(14,6))
    sns.heatmap(pivot, cmap='YlGnBu', cbar_kws={'label': 'arrivals'})
    plt.title('Heatmap: month vs year')
    plt.ylabel('Month')
    plt.savefig(os.path.join(PLOTS_DIR, 'heatmap_month_year.png'), bbox_inches='tight')
    plt.close()


def lstm_ready_plots(ts, look_back=12, test_size=0.2):
    arr = ts.values.reshape(-1,1).astype(float)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(arr)
    scaled_series = pd.Series(scaled.flatten(), index=ts.index, name='scaled')

    # plot scaled series
    plt.figure(figsize=(14,5))
    scaled_series.plot(title='Scaled series for LSTM (0-1)')
    plt.savefig(os.path.join(PLOTS_DIR, 'scaled_series.png'), bbox_inches='tight')
    plt.close()

    # create supervised samples and show a few example input windows
    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i+look_back].flatten())
        y.append(scaled[i+look_back, 0])
    X = np.array(X)
    y = np.array(y)

    n_train = int((1 - test_size) * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # plot a few example windows from training set
    plt.figure(figsize=(12,6))
    for i in range(3):
        plt.plot(range(look_back), X_train[i], label=f'input_{i}')
        plt.scatter(look_back, y_train[i], marker='x', s=60, label=f'target_{i}')
    plt.legend()
    plt.title('Example LSTM input windows (scaled) and their targets')
    plt.xlabel('Lag (t-look_back .. t-1)')
    plt.savefig(os.path.join(PLOTS_DIR, 'lstm_input_examples.png'), bbox_inches='tight')
    plt.close()

    return scaler, (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    ts = load_series()
    plot_series(ts)
    plot_decomposition(ts)
    plot_acf_pacf(ts)
    adf_result = adf_test_and_plot(ts)
    ts_diff = differencing_and_plot(ts, d=1)
    plot_acf_pacf(ts_diff.dropna(), lags=48)
    season_boxplot(ts)
    heatmap_month_year(ts)
    scaler, train, test = lstm_ready_plots(ts)

    print('All plots saved to:', os.path.join(os.path.dirname(__file__), '..', 'plots'))
    print('ADF test result p-value:', adf_result['pvalue'])
    print('Completed visualization for ARIMA and LSTM.')
