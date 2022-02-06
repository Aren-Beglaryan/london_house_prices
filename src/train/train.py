import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import TrainingDataset
from src.model import RegressorNet

MODEL_SAVE_PATH = 'final_model.pt'


def train(train_data_path: str, val_data_path: str, epochs: int, n_input: int,
          batch_size: int = 32, print_every: int = 1) -> None:
    """
    Implements training of simple neural net model for the house price prediction problem.
    After training, the best model's weights are saved to the disk.
    Parameters
    ----------
    train_data_path : str
        Path to the train csv file.
    val_data_path : str
        Path to the validation csv file.
    epochs : int
        Number of epochs to train the model.
    n_input : int
        Number of input features.
    batch_size : int, optional
        Batch size.
    print_every : int, optional
        An integer number indicating how often to print intermediate results (e.g. loss) during each epoch.
        An argument equal to k means print the results after processing k batches.
    """
    train_dataset = TrainingDataset(train_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

    val_dataset = TrainingDataset(val_data_path)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

    net = RegressorNet(n_input=n_input, n_output=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    net.to(device)

    nb_steps = len(train_loader)

    running_loss_train = 0
    best_model = None
    best_loss_val = float('inf')

    for e in range(epochs):
        start = time.time()
        net.train()

        train_loss = 0
        for step, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x, y = x.to(device), y.to(device)

            output = net(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            running_loss_train += loss.item()

            if (step + 1) % print_every == 0:
                net.eval()
                with torch.no_grad():
                    print(
                        "Epoch: {}/{}".format(e + 1, epochs),
                        "Step: {}/{}".format(step + 1, nb_steps),
                        "RMSE Loss: {:.4f}".format(math.sqrt(running_loss_train / print_every)),
                        "{:.3f} s/{} steps".format((time.time() - start), print_every)
                    )
                    running_loss_train = 0
                    start = time.time()

        net.eval()
        with torch.no_grad():
            val_loss = 0
            for step, (x, y) in enumerate(val_loader):
                if torch.cuda.is_available():
                    x, y = x.to(device), y.to(device)

                output = net(x)
                loss = criterion(output, y)
                val_loss += loss.item()

            print(
                "Train loss (RMSE): {:.4f}".format(math.sqrt(train_loss / len(train_loader))),
                "Val loss (RMSE): {:.4f}".format(math.sqrt(val_loss / len(val_loader))),
            )

            if val_loss < best_loss_val:  # Keep the best model
                best_model = copy.deepcopy(net)
                best_loss_val = val_loss

    # Save the best model
    torch.save(best_model.state_dict(), MODEL_SAVE_PATH)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',
                        type=str,
                        default=r"C:\Users\Sololearn\Desktop\london_house_price_prediction\data_files\train.csv",
                        help='Path to the train csv file.')
    parser.add_argument('--val_data_path',
                        type=str,
                        default=r"C:\Users\Sololearn\Desktop\london_house_price_prediction\data_files\val.csv",
                        help='Path to the validation csv file.')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='Number of epochs to train the model')
    parser.add_argument('--n_input',
                        type=int,
                        default=130,
                        help='Number of input features.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size')
    parser.add_argument('--print_every',
                        type=int,
                        default=1,
                        help='Indicates how often to print intermediate results (e.g. loss) during each epoch.'
                             'An argument equal to k means print the results after processing k batches.')
    return parser.parse_args()


def main(args):
    train(train_data_path=args.train_data_path,
          val_data_path=args.val_data_path,
          epochs=args.epochs,
          n_input=args.n_input,
          batch_size=args.batch_size,
          print_every=args.print_every)


if __name__ == '__main__':
    args = parse_args()
    main(args)
