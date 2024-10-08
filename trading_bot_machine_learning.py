# 20241001

import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta
import math
from sklearn.tree import DecisionTreeRegressor
import os

os.system('cls')
plt.close('all')

class TradingBot:

    def execute_trade(self,L,principal,price,trading_signal):

        account_amount = np.zeros(L)
        account_amount[0] = principal # the amount of dollars in the account

        buying_power = account_amount[0]

        n_stocks = 0
        bought_sold = np.zeros(L)
        # buy_sell_amount_limit = 1000
        for i in range(L):
            if trading_signal[i] == 1: # buy
                n = math.floor(buying_power/price.iloc[i])
                # if n > buy_sell_amount_limit:
                    # n = buy_sell_amount_limit

                buying_power = buying_power - n * price.iloc[i]
                
                if n > 0:
                    bought_sold[i] = trading_signal[i]

                n_stocks = n_stocks + n
            elif trading_signal[i] == -1: # sell
                n = n_stocks
                # if n > buy_sell_amount_limit:
                    # n = buy_sell_amount_limit

                buying_power = buying_power + n * price.iloc[i]

                if n > 0:
                    bought_sold[i] = trading_signal[i]

                n_stocks = n_stocks - n

            account_amount[i] = buying_power + n_stocks * price.iloc[i]

        return account_amount, bought_sold

    def performance(self,L,principal,price,account_amount,bought_sold,n_years):
        # if buy and hold the entire time
        buying_power = principal
        n_stocks = math.floor(buying_power/price.iloc[0])
        buying_power = buying_power - n_stocks * price.iloc[0]
        account_amount_2 = buying_power + n_stocks * price

        # stock annual grow rate
        print('stock grow: ')
        print( (account_amount_2[L-1] - principal) / principal / n_years )

        # account annual grow rate
        print('account grow: ')
        print( (account_amount[L-1] - principal) / principal / n_years )

        debug_plot = 1
        if debug_plot == 1:
            figure, axis = plt.subplots(3,1)
            
            stock_grow = price/price.iloc[0]
            axis[0].plot(range(L),stock_grow)
            axis[0].grid()        
            buy_id = np.where(bought_sold==1)
            sell_id = np.where(bought_sold==-1)
            axis[0].scatter(buy_id,stock_grow.iloc[buy_id],marker='o',color='g')
            axis[0].scatter(sell_id,stock_grow.iloc[sell_id],marker='o',color='r')
            axis[0].title.set_text('stock grow. green dot: buy signal, red dot: sell signal')

            axis[1].plot(range(L),account_amount/account_amount[0])
            axis[1].grid()
            axis[1].title.set_text('account grow')

            axis[2].plot(range(L),account_amount_2/1e6,color='b')
            axis[2].plot(range(L),account_amount/1e6,color='g')
            axis[2].grid()
            axis[2].title.set_text('b: stock, g: account')
            axis[2].set_ylabel('millions')

            plt.show()

my_trading_bot = TradingBot()

# Load stock history prices
# --------------------------------------------------
end_date = '2024-08-01'
n_years = 5
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*n_years)
# print(start_date)

df = yf.download(tickers='TSLA', start=start_date, end=end_date)
# print(pd.concat([df.head(5), df.tail(5)]))

price = df['Adj Close']
L = len(price)

# take note the buy and sell points by hand, as supervised learning training data
debug_plot = 0
if debug_plot == 1:
    plt.plot(range(L),price)
    plt.show()

buy_id_train = [156, 257, 276, 400, 451, 576, 600, 644, 707, 860]
sell_id_train = [241, 271, 361, 425, 569, 581, 609, 672, 756, 890]

debug_plot = 0
if debug_plot == 1:
    # plot the training data
    plt.plot(range(L),price)
    plt.scatter(buy_id_train,price.iloc[buy_id_train],marker='o',color='g')
    plt.scatter(sell_id_train,price.iloc[sell_id_train],marker='o',color='r')
    plt.title('green dot: buy, red dot: sell')
    plt.show()

debug = 0
if debug == 1:
    # execute the training data to see the performance
    principal = 1000000

    trading_signal = np.zeros(L) # 1: buy, -1: sell
    trading_signal[buy_id_train] = 1
    trading_signal[sell_id_train] = -1

    account_amount, bought_sold = my_trading_bot.execute_trade(L,principal,price,trading_signal)
    my_trading_bot.performance(L,principal,price,account_amount,bought_sold,n_years)

# Calculate features
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
    axis[0].plot(moving_average)
    axis[0].grid()
    axis[1].plot(rsi)
    axis[1].grid()
    plt.show()

# Train a model
# --------------------------------------------------
id_start = max([rsi_buffer,ma_buffer]) # start after the NaNs
id_stop = max(buy_id_train + sell_id_train)

train_X = np.transpose([price.iloc[id_start:id_stop], rsi[id_start:id_stop], moving_average[id_start:id_stop]])

trading_signal = np.zeros(L) # 1: buy, -1: sell
trading_signal[buy_id_train] = 1
trading_signal[sell_id_train] = -1
train_y = trading_signal[id_start:id_stop]

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

# Make predictions: Generate trading signals
# --------------------------------------------------
id_start = id_stop + 1

val_X = np.transpose([price.iloc[id_start:L], rsi[id_start:L], moving_average[id_start:L]])

trading_signal = model.predict(val_X)

debug_plot = 0
if debug_plot == 1:
    s = price.iloc[id_start:L]
    buy_id = np.where(trading_signal==1)
    sell_id = np.where(trading_signal==-1)

    x = np.arange(id_start,L)
    plt.plot(x,s)
    plt.scatter(x[buy_id],s.iloc[buy_id],marker='o',color='g')
    plt.scatter(x[sell_id],s.iloc[sell_id],marker='o',color='r')
    plt.title('green dot: buy, red dot: sell')
    plt.show()

# Performance of the model
# --------------------------------------------------
s = price.iloc[id_start:L]

# remove the sell signals at the beginning
for i in range(len(s)):
    if trading_signal[i] == -1:
        trading_signal[i] = 0
    elif trading_signal[i] == 1:
        break

debug_plot = 0
if debug_plot == 1:    
    buy_id = np.where(trading_signal==1)
    sell_id = np.where(trading_signal==-1)

    x = np.arange(id_start,L)
    plt.plot(x,s)
    plt.scatter(x[buy_id],s.iloc[buy_id],marker='o',color='g')
    plt.scatter(x[sell_id],s.iloc[sell_id],marker='o',color='r')
    plt.title('green dot: buy, red dot: sell')
    plt.show()

# execute the training data to see the performance
L = len(s)
principal = 1000000
price = s
account_amount, bought_sold = my_trading_bot.execute_trade(L,principal,price,trading_signal)

n_years = L/365
my_trading_bot.performance(L,principal,price,account_amount,bought_sold,n_years)
