import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# choose Australia as analysis target
country_gdp_code = 'RGDPNAAUA666NRUG'
country_name = 'Australia'

# set the start and end dates for the data
start_date = '1960-01-01'
end_date = '2022-01-01'

# download the data from FRED using pandas_datareader
gdp = web.DataReader(country_gdp_code, 'fred', start_date, end_date)
log_gdp = np.log(gdp[country_gdp_code])

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
lambdas = {
    'λ=10': 10,
    'λ=100': 100,
    'λ=1600': 1600
}
trends = {}
cycles = {}
for label, lambda_val in lambdas.items():
    cycle, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=lambda_val)
    trends[label] = trend
    cycles[label] = cycle

# graph 1: trend comparison
plt.plot(log_gdp.index, log_gdp, label='Original log GDP')

for label, trend_series in trends.items():
    plt.plot(trend_series.index, trend_series, label=f'Trend ({label})')

plt.title('Log Real GDP (Australia) and Trend Components (HP Filter)')
plt.xlabel('Year')
plt.ylabel('Log Value')
plt.legend()
plt.grid(True)
plt.show()

# graph 2: business cycle
for label, cycle_series in cycles.items():
    plt.plot(cycle_series.index, cycle_series, label='Cycle ({label})')

plt.axhline(0, color='black', linestyle='--', linewidth=0.8) # ゼロライン
plt.title('Cyclical Components of Log Real GDP (Australia) (HP Filter)')
plt.xlabel('Year')
plt.ylabel('Deviation from Trend (Log Value)')
plt.legend()
plt.grid(True)
plt.show()