from dataclasses import dataclass
import argparse

@dataclass
class Config:
    # Paths
    data_dir: str = '.'
    # Training
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 0.0001
    num_epochs: int = 50
    # Model
    input_dim: int = 40
    seq_len: int = 100
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dropout: float = 0.1

def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--data_dir', type=str, default='.')
    # Training 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=50)
    # Model
    parser.add_argument('--input_dim', type=int, default=40)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    return Config(**vars(parser.parse_args()))