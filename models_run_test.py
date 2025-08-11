import numpy as np
from keras import callbacks
from keras import layers
from keras import models
from keras import optimizers
from keras import utils
import keras.api.ops as K
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from pathlib import Path
import pandas as pd
import wandb
import sys
import argparse

from model_manager import make_model_mse, model_logging, ScaleChecker
from model_Conv1D_50 import make_conv1D
from model_BiLSTM import BiLSTM

from preprocessing import TestPreprocessor

import os

window_size = 150
time = '0.005s'
time_value = time[2:-1]  # '0.005'
flag = 1

attack_type_list = ['fabrication', 'fuzz', 'masquderade', 'replay'] # suspension # 35, 17, 35, 4, 35
for attack_type in attack_type_list:
    current_path = './cache/' + attack_type
    test_parquet_files = sorted(list(Path(current_path).glob(f'*{time_value}.parquet'))) # test 단계에서의 sampling time 조정
    for test_file in test_parquet_files:
            print(f"=============({flag}/91)=============[{attack_type}] 유형: test 파일 - <{test_file}>========================")
            df_test = pd.read_parquet(test_file)
            print(f"min-max scaling 여부: {ScaleChecker(df_test)}") # min-max scaling 되어있는지 확인

            df_test_label = df_test[['label']].copy() # Time (index column), label
            df_test_X = df_test.drop('label', axis=1) # Time (index column), 107 features
            gt_ = df_test_label.rolling(window_size).max().dropna() # preprocessing 단계에 없고 여기서 처리해주어야함
            # print(df_test.shape[0] - window_size + 1)
            # print(gt_.shape[0])
            assert df_test_X.shape[0] - window_size + 1 == gt_.shape[0], '제~~~~~~~~~~발 이러지 말어요'
            assert (df_test_X.iloc[-gt_.shape[0]:].index == gt_.index).all(), '제발'
            # test = TimeseriesGenerator(data=df_test_X.to_numpy(), length=window_size, label = gt_)
            print(f"0.005s 기준으로 전처리한 전체 데이터프레임: {df_test.shape}")
            print(f"위의 df에 대해 label 컬럼을 rolling해가면서 만들어진 binary data: {gt_.shape}")
            print(f"[원시 df에 대한 info]\ntest_shape: {df_test.shape}\nnum_anomalies: {int(gt_.sum().iloc[0])}\nanomaly_ratio: {float(gt_.mean().iloc[0])}")

            print(f"time sampling 후 인덱스컬럼 1번째 값: {df_test.index[0]}")
            print(f"time sampling 후 인덱스컬럼 2번째 값: {df_test.index[1]}")
            print(f"time sampling 후 인덱스컬럼 3번째 값: {df_test.index[2]}")
            print(f"time sampling 후 인덱스컬럼 4번째 값: {df_test.index[3]}")
            print(f"time sampling 후 인덱스컬럼 149번째 값: {df_test.index[148]}")
            print(f"time sampling 후 인덱스컬럼 150번째 값: {df_test.index[149]}")
            print(f"time sampling 후 인덱스컬럼 151번째 값: {df_test.index[150]}")
            print(f"gt_만든 후 인덱스컬럼의 첫 값: {gt_.index[0]}")
            
            # save_dir = './cache/gt'
            # basename = os.path.basename(test_file)  # 'test_fuzz_rate=0070_005.parquet'
            # filename_wo_ext = os.path.splitext(basename)[0]  # 'test_fuzz_rate=0070_005'
            # save_path = os.path.join(save_dir, filename_wo_ext + '.csv')
            # gt_.to_csv(save_path, index=True)

            flag += 1