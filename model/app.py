import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from model.config import config
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from model.plot import PlotData
from model.GRU import GRUModel
from model.LSTM import LSTMModel
from model.rnn import RNNModel
import plotly.express as px
import plotly.graph_objects as go
import ccxt
import json
from datetime import datetime, date, timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CryptoModel:
    def __init__(self):
        # self.createCsv('eth_usd_5min','ETH/USDT')
        self.loadData("data/btc_usd_1day.csv")

    def createCsv(self, csvName, symbol):

        exchange = ccxt.binance()
        from_ts = exchange.parse8601('2023-01-10 00:00:00')
        res = []
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', since=from_ts, limit=1000)
        res.extend(ohlcv)
        while True:
            from_ts = ohlcv[-1][0]
            ohlcv = exchange.fetch_ohlcv(symbol, '5m', since=from_ts, limit=1000)
            res.extend(ohlcv)
            print(len(ohlcv), len(res))
            if len(ohlcv) != 1000:
                res.extend(ohlcv)
                break

        # close_time, quote_asset_volume, number_of_trades,
        # taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore
        df = pd.DataFrame(res, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = [datetime.fromtimestamp(float(time) / 1000) for time in df['time']]
        df['open'] = df['open'].astype(np.float64)
        df['high'] = df['high'].astype(np.float64)
        df['low'] = df['low'].astype(np.float64)
        df['close'] = df['close'].astype(np.float64)
        df['volume'] = df['volume'].astype(np.float64)
        df.set_index('time', inplace=True)
        df.to_csv(f'data/{csvName}.csv')

    def loadData(self, symbol):
        df = pd.read_csv(symbol, sep=',')
        # df = pd.read_csv("./model/binance-dataset.csv", sep=',')
        df = df[["open", "time"]]
        df.set_index("time", inplace=True)
        df.index = pd.to_datetime(df.index)
        input_dim = 60
        df_copy = df.copy()
        df = self.generate_time_lags(df, input_dim)
        df = (
            df
            .assign(minute=df.index.minute)
            .assign(hour=df.index.hour)
            .assign(day=df.index.day)
            .assign(month=df.index.month)
            .assign(day_of_week=df.index.dayofweek)
        )
        df.drop(columns=["month"], inplace=True)
        df = self.generate_cyclical_features(df, 'minute', 60, 0)
        df = self.generate_cyclical_features(df, 'hour', 24, 0)
        df = self.generate_cyclical_features(df, 'day', 31, 0)
        df = self.generate_cyclical_features(df, 'day_of_week', 7, 0)

        index = df.index
        df.reset_index(drop=True, inplace=True)
        X = df.loc[:, df.columns != "open"]
        y = df.loc[:, df.columns == "open"]
        plt.plot(y)
        model_params = {'input_dim': len(X.columns),
                        'hidden_dim': config.HIDDEN_DIM,
                        'layer_dim': config.LAYER_DIM,
                        'output_dim': config.OUTPUT_DIM,
                        'dropout_prob': config.DROPOUT,
                        'device': device}

        model = self.get_model('lstm', model_params)
        model = model.to(device)

        scaler = self.get_scaler('minmax')
        tss = TimeSeriesSplit(n_splits=config.FOLDS, max_train_size=None, test_size=None, gap=0)

        train_losses, validation_losses, oof, y_trues = self.train_function(X, y, model, tss, model_params)
        p = PlotData(oof, train_losses, validation_losses, y_trues)
        oof, y_trues = p.return_values()
        torch.save(model.state_dict(), "model/algo_trade.pth")
        plot_df = pd.DataFrame([oof, y_trues]).T
        plot_df.columns = ["oof", "y_true"]
        df.fillna(0)
        plot_df["date"] = index[0:66280]
        plot_df = plot_df[0:66280]
        plot_df.sort_values(by="date", inplace=True)

        fig = go.Figure()

        fig1 = fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["oof"],
                name="Predicted",  # LINE LEGEND
                marker=dict(color="#DC143C"),  # LINE COLOR
            )
        )
        fig2 = fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["y_true"],
                name="Actual",  # LINE LEGEND
                marker=dict(color="#ecc257"),  # LINE COLOR
            )
        )

        fig.update_layout(
            title_text="Predicted vs Actual BTC/USDT Price",
            template="plotly_dark",
            title_font_color="#cf7200",  # TITLE FONT COLOR
            xaxis=dict(color="#cf7200"),  # X AXIS COLOR
            yaxis=dict(color="#cf7200")  # Y AXIS COLOR
        )
        fig.show()

    def get_scaler(self, scaler):
        scalers = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler(),
        }
        return scalers.get(scaler.lower())

    def get_model(self, model, model_params):
        models = {
            "rnn": RNNModel,
            "lstm": LSTMModel,
            "gru": GRUModel,
        }
        return models.get(model.lower())(**model_params)

    def generate_time_lags(self, df, n_lags):
        df_n = df.copy()
        for n in range(1, n_lags + 1):
            df_n[f"lag{n}"] = df_n["open"].shift(n)
        df_n = df_n.iloc[n_lags:]
        return df_n

    def generate_cyclical_features(self, df, col_name, period, start_num=0):
        kwargs = {
            f'sin_{col_name}': lambda x: np.sin(2 * np.pi * (df[col_name] - start_num) / period),
            f'cos_{col_name}': lambda x: np.cos(2 * np.pi * (df[col_name] - start_num) / period)
        }
        return df.assign(**kwargs).drop(columns=[col_name])

    def train_function(self, X, y, model, tss, model_params):

        # Out of Fold Predictions
        oof = []
        y_trues = []
        # Define MinMaxScaler()
        scaler = self.get_scaler("minmax")
        train_losses = []
        validation_losses = []

        for train_index, valid_index in tss.split(X):
            # Initiate the model
            model = model

            # Create optimizer. Check ReadMe.md for more information.
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.LEARNING_RATE,
                                         weight_decay=config.WEIGHT_DECAY)

            # Create scheduler. Check ReadMe.md for more information.
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          mode='max',
                                          patience=config.LR_PATIENCE,
                                          verbose=False,
                                          factor=config.LR_FACTOR)

            # Create Loss. Check ReadMe.md for more information.
            criterion = nn.MSELoss(reduction="mean")

            # print("Train size:", len(train_index), "Test size:", len(valid_index)), print("\n")
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            # Scale features
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
            y_train = scaler.fit_transform(y_train)
            y_valid = scaler.transform(y_valid)

            X_train = torch.Tensor(X_train)
            X_valid = torch.Tensor(X_valid)
            y_train = torch.Tensor(y_train)
            y_valid = torch.Tensor(y_valid)

            train = TensorDataset(X_train, y_train)
            val = TensorDataset(X_valid, y_valid)

            train_loader = DataLoader(train,
                                      batch_size=config.BATCH_SIZE_TRAIN,
                                      shuffle=False,
                                      drop_last=True)
            val_loader = DataLoader(val,
                                    batch_size=config.BATCH_SIZE_VALIDATION,
                                    shuffle=False,
                                    drop_last=True)

            # === EPOCHS ===
            epochs = config.EPOCHS
            for epoch in range(epochs):
                batch_train_losses = []
                batch_val_losses = []
                start_time = time.time()

                # === TRAIN ===
                # Sets the module in training mode.
                model.train()

                # === Iterate over batches ===
                for x_train_batch, y_train_batch in train_loader:
                    x_train_batch = x_train_batch.view([config.BATCH_SIZE_TRAIN, -1, model_params["input_dim"]]).to(
                        device)
                    y_true = y_train_batch.to(device)
                    # Clear gradients first; very important, usually done BEFORE prediction
                    optimizer.zero_grad()
                    # Log Probabilities & Backpropagation
                    y_pred = model(x_train_batch)
                    loss = criterion(y_true, y_pred)
                    loss.backward()
                    optimizer.step()
                    batch_train_losses.append(loss.item())

                training_loss = np.mean(batch_train_losses)
                train_losses.append(training_loss)

                # === EVAL ===
                # Sets the model in evaluation mode
                model.eval()
                # Disables gradients (we need to be sure no optimization happens)
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.view([config.BATCH_SIZE_VALIDATION, -1, model_params["input_dim"]]).to(device)
                        y_true = y_val.to(device)
                        y_pred = model(x_val)
                        val_loss = criterion(y_true, y_pred).item()
                        batch_val_losses.append(val_loss)
                    validation_loss = np.mean(batch_val_losses)
                    validation_losses.append(validation_loss)

                # Compute time on Train + Eval
                duration = str(timedelta(seconds=time.time() - start_time))[:7]

                # Print status
                if epoch == 49:
                    print(
                        f"[{epoch}/{epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                    )
                # Update scheduler (for learning_rate)
                scheduler.step(validation_loss)

            # === Out of Fold predictions ===
            # Sets the model in evaluation mode
            model.eval()
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.view([config.BATCH_SIZE_VALIDATION, -1, model_params["input_dim"]]).to(device)
                    y_true = y_val.to(device)
                    y_pred = model(x_val)
                    oof.append(scaler.inverse_transform(y_pred.cpu().numpy()))
            y_trues.append(scaler.inverse_transform(y_valid.cpu().numpy()))

        return train_losses, validation_losses, oof, y_trues
