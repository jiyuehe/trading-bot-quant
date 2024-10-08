# 20240827

import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta
import math

plt.close('all')

# Load stock history prices
# --------------------------------------------------
end_date = '2024-08-01'
n_years = 1
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*n_years)
# print(start_date)

df = yf.download(tickers='TSLA', start=start_date, end=end_date)
# print(pd.concat([df.head(5), df.tail(5)]))

price = df['Adj Close']

debug_plot = 0
if debug_plot == 1:
    plt.plot(price)
    plt.show()

# Calculate indicators
# --------------------------------------------------
rsi_buffer = 14
rsi = pandas_ta.rsi(close=price, length=rsi_buffer)
# print(rsi)

ma_buffer = 14
moving_average = pandas_ta.sma(close=price, length=ma_buffer) # moving average
# print(rsi)

debug_plot = 0
if debug_plot == 1:
    figure, axis = plt.subplots(2,1)
    axis[0].plot(price)
    axis[0].grid()
    axis[1].plot(rsi)
    axis[1].grid()
    plt.show()

    # figure, axis = plt.subplots(2,1)
    plt.plot(price)
    plt.grid()
    plt.plot(moving_average)
    plt.show()

# Generate trading signals
# --------------------------------------------------
L = len(price)

trading_signal = np.zeros(L) # 1: buy, -1: sell

rsi_high = 75
rsi_low = 40
ma_threshold = 1/100 # percentage
for i in range(L):
    if moving_average.iloc[i] - price.iloc[i] > moving_average.iloc[i]*ma_threshold \
        and rsi.iloc[i] < rsi_low :
        trading_signal[i] = 1 # buy
    elif price.iloc[i] - moving_average.iloc[i] > moving_average.iloc[i]*ma_threshold \
        and rsi.iloc[i] > rsi_high:
        trading_signal[i] = -1 # sell

debug_plot = 0
if debug_plot == 1:
    plt.plot(trading_signal)
    plt.show()

# Execute the trade
# --------------------------------------------------
account_amount = np.zeros(L)
principal = 1000000
account_amount[0] = principal # the amount of dollars in the account

buying_power = account_amount[0]

n_stocks = 0
bought_sold = np.zeros(L)
buy_sell_amount_limit = 1000
for i in range(L):
    if trading_signal[i] == 1: # buy
        n = math.floor(buying_power/price.iloc[i])
        if n > buy_sell_amount_limit:
            n = buy_sell_amount_limit

        buying_power = buying_power - n * price.iloc[i]
        
        if n > 0:
            bought_sold[i] = trading_signal[i]

        n_stocks = n_stocks + n
    elif trading_signal[i] == -1: # sell
        n = n_stocks
        if n > buy_sell_amount_limit:
            n = buy_sell_amount_limit

        buying_power = buying_power + n * price.iloc[i]

        if n > 0:
            bought_sold[i] = trading_signal[i]

        n_stocks = n_stocks - n

    account_amount[i] = buying_power + n_stocks * price.iloc[i]

# Comparison
# --------------------------------------------------
# if buy and hold the entire time
buying_power = principal
n_stocks = math.floor(buying_power/price.iloc[0])
buying_power = buying_power - n_stocks * price.iloc[0]
account_amount_2 = buying_power + n_stocks * price.iloc[L-1]

# compare to the stock itself
print( (account_amount[L-1] - account_amount_2) / principal / n_years )

# compare to the principal
print( (account_amount[L-1] - principal) / principal / n_years )

debug_plot = 1
if debug_plot == 1:
    figure, axis = plt.subplots(3,1)

    axis[0].plot(range(L),price)
    axis[0].plot(range(L),moving_average)
    axis[0].grid()

    buy_id = np.where(bought_sold == 1)
    sell_id = np.where(bought_sold == -1)
    axis[0].scatter(buy_id,price.iloc[buy_id],marker='o',color='g')
    axis[0].scatter(sell_id,price.iloc[sell_id],marker='o',color='r')
    axis[0].title.set_text('buy: green dot, sell: red dot')

    rsi_2 = rsi
    rsi_2[np.isnan(rsi_2)] = 50 # a hack to make all plots aligned
    axis[1].plot(range(L),rsi_2)
    axis[1].grid()
    axis[1].title.set_text('RSI')
    
    axis[2].plot(range(L),price/price.iloc[0])
    axis[2].plot(range(L),account_amount/principal)
    axis[2].grid()
    axis[2].title.set_text('stock grow vs trading grow')

    plt.show()
    