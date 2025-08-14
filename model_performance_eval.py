import numpy as np
from keras import callbacks
from keras import layers
from keras import models
from keras import optimizers
from keras import utils
import keras.api.ops as K
from pathlib import Path
import pandas as pd
import wandb
import sys
import argparse
from model_BiLSTM import BiLSTM
from preprocessing import TestPreprocessor
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from models_main import TimeseriesGenerator, MergedTimeseriesGenerator
import os
from model_manager import make_model_mse2


"""
    pseudo code:
    1. 저장된 모델 불러오기
    2. train 데이터셋에 대해 모델을 사용하여 예측 수행 및 l(i)의 집합인 벡터 l 생성 - a
    3. validation 데이터셋에 대해 모델을 사용하여 예측 수행 및 l(i)의 집합인 벡터 l 생성 - b
    4. a를 이용하여 small theta 작업 및 small theta의 집합인 벡터 생성 - c
    5. 길이가 x(107)인 벡터 r을 생성: r(i) = b / c
    6. big theta = percentile(r, q) # 0.95 <= q <= 1.0
    7. big theta를 이용하여 r(i) > big theta인 경우를 이상치로 판단
    """


class model_performance_eval:
    def __init__(self, model_path=None, window_size=150, hidden_space=125, q=0.993):
        self.model_path = model_path
        self.window_size = window_size
        self.hidden_space = hidden_space
        self.q = q
        if model_path is None:
            raise ValueError("모델 경로가 지정되지 않았습니다.")
        print(f"Evaluating model performance of {self.model_path}...")
        print("__init__ function is loading the model...")

        # 모델 로드 (학습 없이 predict만)
        self.model = models.load_model(
            filepath=self.model_path,
            custom_objects={'BiLSTM': BiLSTM}
        )
        self.model_mse = make_model_mse2(self.model)  # MSE 모델 생성
        self.model_mse.summary(expand_nested=True)
        print("Model loaded successfully!")

        print(f"Model loaded successfully from {self.model_path}.")

    def evaluate(self, test_data, gt_, threshold_big_theta=None, train_small_theta_vector=None, train_data=None, validation_data=None):

        if test_data is None:
            raise ValueError("test_data 없음.")

        if threshold_big_theta is None or train_small_theta_vector is None:
            if train_data is None or validation_data is None:
                raise ValueError("train_data and validation_data 없음.")
            threshold_big_theta, train_small_theta_vector = self.prepare(train_data, validation_data)

        test_l_vector = self.loss_vector_l(test_data)
        error_rate_vector = test_l_vector / train_small_theta_vector
        anomaly_prediction_bool = np.max(error_rate_vector, axis=1) > threshold_big_theta
        anomaly_signals = np.argmax(error_rate_vector[anomaly_prediction_bool], axis=1)

        # 평가를 어케 하지? # 교수님 question
        true_anomaly = (gt_ == 1) # 정상:0, 이상:1 -> 정상:false, 이상:true
        test_f1 = f1_score(true_anomaly, anomaly_prediction_bool)
        test_precision = precision_score(true_anomaly, anomaly_prediction_bool)
        test_recall = recall_score(true_anomaly, anomaly_prediction_bool)

        return test_f1, test_precision, test_recall, anomaly_prediction_bool, anomaly_signals

    def prepare(self, train_data, validation_data):

        train_l_vector = self.loss_vector_l(train_data)
        train_small_theta_vector = self.signal_theta(train_l_vector)

        val_l_vector = self.loss_vector_l(validation_data)

        error_rate_vector = val_l_vector / train_small_theta_vector

        threshold_big_theta = np.percentile(np.max(error_rate_vector, axis=1), self.q*100) # (samples, 107) -> (samples,) -> scalar

        print(f"threshold_big_theta: {threshold_big_theta}, train_small_theta_vector: {train_small_theta_vector}")
        return threshold_big_theta, train_small_theta_vector

    def loss_vector_l(self, tg_data, test=False):

        l_vector = self.model_mse.predict(tg_data)
        return l_vector


    def signal_theta(self, l_vector):
        # 분모가 될 small theta 작업을 수행하는 로직 구현
        mean = np.mean(l_vector, axis=0) # l_vector는 (samples, 107) 형태
        std = np.std(l_vector, axis=0)
        small_theta_vector = mean + 3 * std # 3-sigma rule 적용
        return small_theta_vector

    
    def log_results(self):
        # 결과를 WandB에 로깅하는 로직 구현
        pass

if __name__ == "__main__":
    # model
    model_name = './models/autoencoder_model_250805_1912.keras'

    print(os.getcwd())
    
    # data
    time_value = '005' # time = '0.005s'
    window_size = 150
    
    # Load and prepare training data
    df1_name = './cache/df1_merged_' + time_value + '.parquet'
    df2_name = './cache/df2_merged_' + time_value + '.parquet'
    df3_name = './cache/df3_merged_' + time_value + '.parquet'
    df4_name = './cache/df4_merged_' + time_value + '.parquet'
    
    df1 = pd.read_parquet(df1_name)
    df2 = pd.read_parquet(df2_name)
    df3 = pd.read_parquet(df3_name)
    df4 = pd.read_parquet(df4_name)

    tg1 = TimeseriesGenerator(data=df1.to_numpy(), length=window_size, shuffle=True)
    tg2 = TimeseriesGenerator(data=df2.to_numpy(), length=window_size, shuffle=True)
    tg3 = TimeseriesGenerator(data=df3.to_numpy(), length=window_size, shuffle=True)
    tg4 = TimeseriesGenerator(data=df4.to_numpy(), length=window_size, shuffle=True)
    
    train = MergedTimeseriesGenerator([tg1, tg2, tg3, tg4], shuffle=True) # , tg2, tg3, tg4
    # train = [tg1, tg2, tg3, tg4]

    # Load and prepare validation data
    df_val_name = './cache/df5_merged_' + time_value + '.parquet'
    df_val = pd.read_parquet(df_val_name)
    validation = TimeseriesGenerator(data=df_val.to_numpy(), length=window_size, shuffle=True)

    # # Load and prepare test data
    # df_test_one_name = './cache/fabrication/test_fabrication_aid=316_' + time_value + '.parquet'
    # df_test = pd.read_parquet(df_test_one_name)
    # df_test_label = df_test[['label']].copy()
    # df_test = df_test.drop('label', axis=1)
    
    # # Preprocess ground truth labels
    # gt_ = df_test_label.rolling(window_size).max().dropna()
    # # The `gt_` label needs to be reshaped to a 1D array to match the `y_true` format for scikit-learn metrics.
    # gt_ = gt_.to_numpy().flatten()
    
    # assert df_test.shape[0] - window_size + 1 == gt_.shape[0], 'Ground truth label count does not match test data sequence count.'
    
    # # Create TimeseriesGenerator for test data
    # test = TimeseriesGenerator(data=df_test.to_numpy(), length=window_size, shuffle=False) # shuffle=False for reproducible evaluation

    model_eval = model_performance_eval(model_path=model_name)

    # Step 1-6: Prepare thresholds using training and validation data
    print("Preparing thresholds with training and validation data...")
    threshold_big_theta, train_small_theta_vector = model_eval.prepare(
        train_data=train,
        validation_data=validation
    )
    print(f"Calculated Big Theta (Threshold): {threshold_big_theta}")

    flag = 1
    # train 및 validation 완료, test 시작
    test_results_for_table = []
    test_pp = TestPreprocessor()
    attack_type_list = ['fabrication', 'fuzz', 'masquderade', 'replay'] # suspension # 35, 17, 35, 4, 35
    for attack_type in attack_type_list:
        current_path = './cache/' + attack_type
        test_parquet_files = sorted(list(Path(current_path).glob(f'*{time_value}.parquet'))) # test 단계에서의 sampling time 조정
        for test_file in test_parquet_files:
                print(f"=============({flag}/91)=============[{attack_type}] 유형: test 파일 - <{test_file}>========================")
                df_test = pd.read_parquet(test_file)
                # ScaleChecker(df_test) # min-max scaling 되어있는지 확인
                df_test_label = df_test[['label']].copy() # Time (index column), label
                df_test = df_test.drop('label', axis=1) # Time (index column), 107 features
                gt_ = df_test_label.rolling(window_size).max().dropna() # preprocessing 단계에 없고 여기서 처리해주어야함
                # print(df_test.shape[0] - window_size + 1)
                # print(gt_.shape[0])
                assert df_test.shape[0] - window_size + 1 == gt_.shape[0], '제~~~~~~~~~~발 이러지 말어요'
                test = TimeseriesGenerator(data=df_test.to_numpy(), length=window_size, label = gt_)

                # Step 7: Evaluate the model on the test dataset
                print("Evaluating model performance on test data...")
                test_f1, test_precision, test_recall, anomaly_prediction_bool, anomaly_signals = model_eval.evaluate(
                    test_data=test,
                    gt_=gt_,
                    threshold_big_theta=threshold_big_theta,
                    train_small_theta_vector=train_small_theta_vector
                )

                # _, sub_identifier = test_pp.get_attack_type_from_filename(test_file.name) # 파일 이름 얻기
                # Log and print results
                print("--- Evaluation Results ---")
                print(f"F1 Score: {test_f1:.4f}")
                print(f"Precision: {test_precision:.4f}")
                print(f"Recall: {test_recall:.4f}")
                print("--------------------------")