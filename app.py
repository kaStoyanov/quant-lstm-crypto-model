import torch
import ccxt

from model.GRU import GRUModel
from model.LSTM import LSTMModel
from model.app import CryptoModel
from model.config import config
from model.rnn import RNNModel

app = CryptoModel()
exchange = ccxt.binance()


def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res = []
data = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=1)
print(data)
model_params = {
    'input_dim': len(data),
    'hidden_dim': config.HIDDEN_DIM,
    'layer_dim': config.LAYER_DIM,
    'output_dim': config.OUTPUT_DIM,
    'dropout_prob': config.DROPOUT,
    'device': device
}
model = get_model('lstm', model_params)
model = model.to(device)
model.load_state_dict(torch.load("model/algo_trade.pth"))
model.eval()

print(model(data))
