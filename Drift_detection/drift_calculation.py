# https://stackoverflow.com/questions/33578787/how-to-interpolate-a-single-column-of-data-to-match-the-shape-of-another-column

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.stats import ks_2samp, anderson_ksamp, chi2_contingency, probplot

# Load your EUR/USD historical data
data = pd.read_csv('../../EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(10000)
data = data['close']
train_data = data[2000:3000]
test_data = data[2000:2500]

# Function to calculate log returns
def calculate_log_returns(prices):
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()

# Calculate log returns for both datasets
log_returns_train = calculate_log_returns(train_data)
log_returns_test = calculate_log_returns(test_data)

# Do interpolation to make the lengths of the two series equal
f = interp1d(np.arange(len(log_returns_train)), log_returns_train, kind='linear', fill_value="extrapolate")
log_returns_train_interpolated = f(np.linspace(0, len(log_returns_train)-1, len(log_returns_test)))

# Define methods for KDE plots
methods = [
    ('Before Interpolation', log_returns_train, log_returns_test),
    ('After Interpolation', log_returns_train_interpolated, log_returns_test)
]

# Create a single figure for all plots
fig, axs = plt.subplots(len(methods), 3, figsize=(18, 12))

for i, (method_name, ref_data, curr_data) in enumerate(methods):
    # Line plots
    axs[i, 0].plot(ref_data, label='Train')
    axs[i, 0].plot(curr_data, label='Test')
    axs[i, 0].set_title(f'Log Returns: Train vs Test ({method_name})')
    axs[i, 0].set_xlabel('Time')
    axs[i, 0].set_ylabel('Log Returns')
    axs[i, 0].legend()

    # Histograms
    axs[i, 1].hist(ref_data, bins=30, alpha=0.5, label='Train')
    axs[i, 1].hist(curr_data, bins=30, alpha=0.5, label='Test')
    axs[i, 1].set_title(f'Histogram: Train vs Test ({method_name})')
    axs[i, 1].set_xlabel('Log Returns')
    axs[i, 1].set_ylabel('Frequency')
    axs[i, 1].legend()

    # KDE plots
    sns.kdeplot(ref_data, ax=axs[i, 2], label='Train', color='blue', fill=True)
    sns.kdeplot(curr_data, ax=axs[i, 2], label='Test', color='orange', fill=True)
    axs[i, 2].set_title(f'KDE: Train vs Test ({method_name})')
    axs[i, 2].set_xlabel('Log Returns')
    axs[i, 2].set_ylabel('Density')
    axs[i, 2].legend()

plt.tight_layout()
plt.show()

# Statistical tests
for method_name, ref_data, curr_data in methods:
    print(f"\nStatistical tests for {method_name}:")
    
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p_value = ks_2samp(ref_data, curr_data)
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, p-value={ks_p_value}")
    
    # Anderson-Darling Test
    ad_result = anderson_ksamp([ref_data, curr_data])
    print(f"Anderson-Darling Test: Statistic={ad_result.statistic}, p-value={ad_result.significance_level}")
    
    # Chi-Square Test
    hist_ref, _ = np.histogram(ref_data, bins=30)
    hist_curr, _ = np.histogram(curr_data, bins=30)
    chi2_stat, chi2_p_value, _, _ = chi2_contingency([hist_ref, hist_curr])
    print(f"Chi-Square Test: Statistic={chi2_stat}, p-value={chi2_p_value}")
    
    # Descriptive Statistics
    print(f"Descriptive Statistics for Reference Data - {method_name}:")
    print(ref_data.describe())
    print(f"Descriptive Statistics for Current Data - {method_name}:")
    print(curr_data.describe())
    
    # Q-Q Plot
    plt.figure(figsize=(6, 6))
    probplot(ref_data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of Reference Data - {method_name}')
    plt.show()
    
    plt.figure(figsize=(6, 6))
    probplot(curr_data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of Current Data - {method_name}')
    plt.show()