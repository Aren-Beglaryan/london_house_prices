import pandas as pd

import torch

from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    """Training Dataset class for house price prediction"""

    TARGET_COLUMN = 'price'

    def __init__(self, data_csv_path):
        self.df = pd.read_csv(data_csv_path).sample(frac=1).reset_index(drop=True)

    def __getitem__(self, indx):
        df_row = self.df.iloc[indx]

        x = torch.from_numpy(df_row.drop(self.TARGET_COLUMN).values).float()
        y = torch.tensor([float(df_row[self.TARGET_COLUMN])])

        return x, y

    def __len__(self):
        return len(self.df)


