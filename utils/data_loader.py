import pandas as pd


def load_manifest(csv_path):
    return pd.read_csv(csv_path)