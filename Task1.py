import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np


# define countries for analysis
countries = {
    'Australia': {'gdp_code': 'RGDPNAAUA666NRUG', 'name': 'Australia'},
    'Japan': {'gdp_code': 'RGDPNAJPA666NRUG', 'name': 'Japan'}
}

# set the start and end dates for the data
start_date = '1960-01-01'
end_date = '2022-01-01'

# lists
lambdas = {'λ=1600': 1600}
all_log_gdp = {}
all_trends = {}
all_cycles = {}

for country_name, country_info in countries.items():
    gdp_code = country_info['gdp_code']

    # download the data from FRED using pandas_datareader
    gdp = web.DataReader(gdp_code, 'fred', start_date, end_date)
    log_gdp = np.log(gdp[gdp_code])
    all_log_gdp[country_name] = log_gdp

    # apply a Hodrick-Prescott filter to the data to extract the cyclical component
    trends = {}
    cycles = {}
    for label, lambda_val in lambdas.items():
        cycle, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=lambda_val)
        trends[label] = trend
        cycles[label] = cycle
    all_trends[country_name] = trends
    all_cycles[country_name] = cycles

# calculate standard deviations
print("\n--- Standard Deviations ---")
std_devs = {}
for country_name, cycles_data in all_cycles.items():
    std_dev = cycles_data['λ=1600'].std()
    std_devs[country_name] = std_dev
    print(f"{country_name} Standard Deviation (λ=1600): {std_dev:.4f}")

# calculate correlations
print("\n--- Correlation Coefficients ---")
australia_cycle = all_cycles['Australia']['λ=1600'].dropna()
japan_cycle = all_cycles['Japan']['λ=1600'].dropna()
common_index = australia_cycle.index.intersection(japan_cycle.index)
correlation = australia_cycle.loc[common_index].corr(japan_cycle.loc[common_index])
print(f"Correlation Coefficient (λ=1600): {correlation:.4f}")

# generate graph
plt.figure(figsize=(12, 7))
for country_key, country_info in countries.items():
    display_name = country_info['name']
    plt.plot(all_cycles[display_name]['λ=1600'].index, all_cycles[display_name]['λ=1600'], label=f'{display_name} Cycle (λ=1600)', linewidth=2)

plt.axhline(0, color='black', linestyle='--', linewidth=0.8) # Zero line
plt.title('Cyclical Components of Log Real GDP (λ=1600)')
plt.xlabel('Year')
plt.ylabel('Deviation from Trend (Log Value)')
plt.legend()
plt.grid(True)
plt.show()