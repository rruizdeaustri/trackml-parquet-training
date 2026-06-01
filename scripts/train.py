#!/usr/bin/env python3
import argparse
from tracking_train.train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the tracking classifier")
    parser.add_argument("config", help="Path to TOML config")
    args = parser.parse_args()
    main(args.config)
