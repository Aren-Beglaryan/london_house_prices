import pandas as pd
import torch
from typing import Dict
from src.model import RegressorNet
from pandas import Timestamp
from src.data.create_dataset import preprocess_data, one_hot_encode_transform, normalize


class Predictor:
    def __init__(self, model_path: str):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        net = RegressorNet(n_input=130)
        net.to(device)

        # if torch.cuda.is_available():
        #     net.load_state_dict(torch.load(model_path))
        # else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        net.eval()

        self.net = net

    def infer(self, payload: Dict):
        processed_sample = self.preprocess(payload)
        pred = self.net(processed_sample)

        return self.postprocess(pred)

    @staticmethod
    def preprocess(payload: Dict):
        sample = pd.DataFrame([payload.values()], columns=payload.keys())
        sample['date'] = pd.to_datetime(sample['date'])
        sample = preprocess_data(sample)
        sample = one_hot_encode_transform(sample)
        sample = normalize(sample)

        return torch.from_numpy(sample.values).float()

    @staticmethod
    def postprocess(x):
        return {'prediction': float(x.detach().numpy()[0][0])}


if __name__=="__main__":
    predictor = Predictor(r'C:\Users\Sololearn\Desktop\london_house_price_prediction\src\train\final_model.pt')
    payload = {'address': 'Flat 3, Warwick Apartments, 132, Cable Street, London, Greater London E1 8NU',
             'type': 'Flat',
             'bedrooms': 3,
             'latitude': 51.51092,
             'longitude': -0.0625,
             'area': 'E1',
             'tenure': 'Leasehold',
             'is_newbuild': 1,
             'date': Timestamp('2014-01-28 00:00:00+0000', tz='UTC')}

    print(predictor.infer(payload))