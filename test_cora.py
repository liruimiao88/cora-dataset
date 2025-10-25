import os
import torch
from cora_dataset import CoraDataset

def main():
    print("Starting test...")
    train_data = CoraDataset('./data', train=True)
    test_data = CoraDataset('./data', train=False)
    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")
    data, label = train_data[0]
    print(f"Data shape: {data.shape}")
    print(f"Label: {label}")
    print("Success!")

if __name__ == "__main__":
    main()