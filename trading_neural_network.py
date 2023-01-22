'''
In this example, a neural network architecture is created using the Sequential class and the Dense layer from Keras library. The code uses the compile method to specify the optimizer and the loss function. Then, the model is trained using the fit method for 100 epochs with a batch size of 32 and not showing the progress.

It's worth noting that the neural network predictions require the input data to be normalized before the training process, in this case it's done with MinMaxScaler from sklearn and then the same transformation is applied to the indicators before the prediction step.
'''

#Trading bot using feedforward Neural Network

# Importing the necessary libraries
import ccxt
import pandas as pd
import numpy as np
from ta import *
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Collecting historical market data from Binance
exchange = ccxt.binance({
    'rateLimit': 2000,
    'enableRateLimit': True,
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET',
    'options': {
        'adjustForTimeDifference': True,
    },
})
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d')
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Preprocessing the data
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Adding the indicators
df['macd'] = macd(df['close'])
df['rsi'] = rsi(df['close'])
df['cci'] = cci(df['high'], df['low'], df['close'])
df['dx'] = dx(df['high'], df['low'], df['close'])
df['mma'] = sma(df['close'])

# Splitting the data into training and testing sets
X = df.drop(['close'], axis=1)
y = df['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the neural network
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Training the neural network
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Evaluating the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Model MSE: {score:.2f}')

# Implementing the trading bot
while True:
    # Collecting the current market data
    ticker = exchange.fetch_ticker('BTC/USDT')
    current_price = ticker['last']
    indicators = indicators(current_price)
    order_book = exchange.fetch_order_book('BTC/USDT')

    # Adding the order book information to the indicators
    indicators.append(order_book['bids'][0][0])
    indicators.append(order_book['asks'][0][0])

    # Making a prediction
    indicators = scaler.transform([indicators])
    prediction = model.predict(indicators)
    # Executing the trade
    if prediction > current_price:
        exchange.create_limit_buy_order('BTC/USDT', 0.01, current_price*1.01)
    elif prediction < current_price:
        exchange.create_limit_sell_order('BTC/USDT', 0.01, current_price*0.99)
    # Wait for some time
    time.sleep(600)


'''
The code above is using a type of neural network called a feedforward neural network, also known as a multi-layer perceptron (MLP). It is a type of artificial neural network that is designed to take a set of input features, process them through a series of hidden layers with non-linear activation functions, and output a prediction or decision.

In this specific example, the neural network has 4 layers: an input layer, 2 hidden layers, and an output layer. The input layer has 64 neurons, the first hidden layer has 32 neurons, the second hidden layer has 16 neurons, and the output layer has 1 neuron. Each layer is connected to the next with a set of parameters called weights. The input layer takes in the input data, in this case the indicators and order book information, and the output layer produces the prediction.

The activation functions used in the hidden layers are rectified linear units (ReLU). This activation function is defined as ReLU(x) = max(0, x) which means that it returns the input if it is positive, otherwise, it returns zero. ReLU is a widely used activation function because it is computationally efficient and it helps to alleviate the vanishing gradient problem.

The optimizer used is Adam. Adam is a gradient-based optimization algorithm that adapts the learning rate for each parameter.
The loss function used is mean squared error (MSE) which is a common loss function for regression problems. MSE calculates the average squared difference between the predicted values and the true values.

It's important to note that this is a simple example and there are many other types of neural networks and architectures that can be used for this problem. To choose the best neural network architecture and parameters, it is necessary to test different options and evaluate their performance using a robust evaluation metric.
'''


