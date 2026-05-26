import pandas as pd
import torch
import sys
sys.path.insert(0, '/mnt/c/Users/chenyihao/Documents/GitHub/fluid-shape-pipeline')

df = pd.read_csv('data/wake_fields/index.csv')
print(f'{len(df)} cases')
print(f'shapes: {sorted(df["shape"].unique())}')
print(f'Re: {sorted(df["Re"].unique())}')

from ml.train_wake import parse_args
print('train_wake import OK')

from vision.jepa_encoder import LightweightCNNEncoder
print('jepa_encoder import OK')

print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
