import time
import random
from datetime import datetime
from pathlib import Path
from typing import Literal
import json

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import MulticlassPrecision, MulticlassRecall
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from model import DeepLOB, LOBFormer
from config import parse_args, Config

def fix_all_seeds(seed):
    '''Fixes RNG seeds for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def time_since(base: float, format=None):
    now = time.time()
    elapsed_time = now - base
    if format == 'seconds':
        return elapsed_time
    else:
        return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
def get_datetime_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S")
class Dataset(Dataset):
    def __init__(self, data: np.ndarray, k: int=4, seq_len: int=100, model: Literal['transformer', 'cnn']='transformer'):
        """
        data: np.ndarray
            The raw data to be converted to a dataset.
        k: int
            Index of prediction horizon. 0–4 correspond to steps ahead: [1, 2, 3, 5, 10].
        seq_len: int
            The length of time series window.
        """
        (num_samples, _) = data.shape # (254750, 149) if using Train_Dst_NoAuction_DecPre_CF_7.txt
        data = torch.tensor(data, dtype=torch.float32)
        labels = data[seq_len - 1:, k-5].to(torch.int64) - 1 # Map to 0-based index

        features = torch.zeros(num_samples - seq_len + 1, seq_len, 40)
        _data = data[:, :40]
        for i in range(num_samples - seq_len + 1):
            features[i] = _data[i:i+seq_len, :]

        self.length = num_samples - seq_len + 1
        if model == 'transformer':
            self.features = features
        elif model == 'cnn':
            self.features = features.unsqueeze(1) # Unsqueeze for channel dimension
        self.labels = labels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        writer: SummaryWriter,
        ckpt_dir: Path):
    precision = MulticlassPrecision(num_classes=3, average="macro")
    recall = MulticlassRecall(num_classes=3, average="macro")
    best_valid_loss = np.inf # TODO optimize
    train_losses = []
    valid_losses = []

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        for features, labels in tqdm(train_loader, desc='Training'):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        # Validation
        model.eval()
        valid_loss = 0
        precision.reset()
        recall.reset()
        for features, labels in tqdm(valid_loader, desc='Validation'):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            # Evaluation
            precision.update(outputs, labels)
            recall.update(outputs, labels)
        valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(valid_loss)
        prompt = (
            f'Epoch: {epoch: 3}/{num_epochs} '
            f'Elapsed Time: {time_since(start_time)} '
            f'Train Loss: {train_loss: .4f} '
            f'Valid Loss: {valid_loss: .4f} '
            f'Precision: {precision.compute(): .4f} '
            f'Recall: {recall.compute(): .4f}'
        )
        tqdm.write(prompt)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Precision/valid', precision.compute(), epoch)
        writer.add_scalar('Recall/valid', recall.compute(), epoch)
        
        # Save
        torch.save(model, ckpt_dir / f'ckpt-{epoch}.pth')
        if valid_loss < best_valid_loss:
            torch.save(model, ckpt_dir / 'ckpt-best.pth')
            best_valid_loss = valid_loss
            tqdm.write('Model saved')

def test_model(
        test_loader: DataLoader,
        ckpt_dir: Path
):
    precision = MulticlassPrecision(num_classes=3, average="macro")
    recall = MulticlassRecall(num_classes=3, average="macro")
    # model = torch.load(ckpt_dir / 'ckpt-best.pth', weights_only=False)

    # model.eval()
    # with torch.no_grad():
    #     for features, labels in tqdm(test_loader, desc='Testing'):
    #         features, labels = features.to(device), labels.to(device)
    #         outputs = model(features)

    #         precision.update(outputs, labels)
    #         recall.update(outputs, labels)

    # precision = precision.compute()
    # recall = recall.compute()
    # tqdm.write(f'Precision: {precision: .4f} Recall: {recall: .4f}')

def load_dataset(data_dir: str, batch_size: int, split: Literal['Train', 'Test']):
    if 'Train' in split:
        raw_train_data = np.loadtxt(f'{data_dir}/Train_Dst_NoAuction_DecPre_CF_7.txt', dtype=np.float32)
        split_point = int(raw_train_data.shape[1] * 0.8)
        train_data = raw_train_data[:, :split_point]
        valid_data = raw_train_data[:, split_point:]

        train_dataset = Dataset(train_data.T)
        valid_dataset = Dataset(valid_data.T)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader
    else:
        test_data1 = np.loadtxt(f'{data_dir}/Test_Dst_NoAuction_DecPre_CF_7.txt', dtype=np.float32)
        test_data2 = np.loadtxt(f'{data_dir}/Test_Dst_NoAuction_DecPre_CF_8.txt', dtype=np.float32)
        test_data3 = np.loadtxt(f'{data_dir}/Test_Dst_NoAuction_DecPre_CF_9.txt', dtype=np.float32)
        test_data = np.hstack((test_data1, test_data2, test_data3))

        test_dataset = Dataset(test_data.T)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader

def main(args: Config):
    fix_all_seeds(args.seed)

    # Checkpoints
    if not Path('./checkpoints').exists():
        Path('./checkpoints').mkdir()
    ckpt_dir = Path('./checkpoints') / get_datetime_now() 
    ckpt_dir.mkdir()
    # Save arguments
    with open(f'{ckpt_dir}/args.json', 'w') as f:
        args_dict = {k: str(v) for k, v in vars(args).items()}
        json.dump(args_dict, f)
    # Tensorboard
    writer = SummaryWriter(
        log_dir=ckpt_dir
    )
    # Training Arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LOBFormer(args.input_dim, args.seq_len, args.d_model, args.nhead, args.num_layers, args.dropout)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader, valid_loader = load_dataset(args.data_dir, args.batch_size, 'Train')
    train_model(model, device, criterion, optimizer, train_loader, valid_loader, args.num_epochs, writer, ckpt_dir)

def visualize_dataset(data_dir: str):
    raw_train_data = np.loadtxt(f'{data_dir}/Train_Dst_NoAuction_DecPre_CF_7.txt', dtype=np.float32)
    train_data = raw_train_data[:, :100]


    ask_price_lv1 = train_data[0, :]
    bid_price_lv1 = train_data[2, :]
    mid_price = (ask_price_lv1 + bid_price_lv1) / 2

    labels = train_data[-5, :]
    labels = np.roll(labels, -1)


    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot labels
    for i, label in enumerate(labels):
        if label == 1:
            ax.axvspan(i, i + 1, color='red', alpha=0.3)
        elif label == 3:
            ax.axvspan(i, i + 1, color='green', alpha=0.3)

    plt.plot(mid_price, label='Mid Price (Normalized)')
    # plt.plot(ask_price_lv1, label='Ask Price L1 (Norm)', linestyle='--')
    # plt.plot(bid_price_lv1, label='Bid Price L1 (Norm)', linestyle='--')
    ax.set_title(f"Relative Price Trend")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Normalized Price")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)