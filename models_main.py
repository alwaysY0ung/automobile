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

class TimeseriesGenerator(utils.Sequence):
    def __init__(self, data, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False,
                 reverse=False, batch_size=128, label=None):
        super().__init__()
        self.data = data
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data)
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.label = label if label is None else np.array(label)
        if self.start_index > self.end_index:
            raise ValueError(
                "`start_index+length=%i > end_index=%i` "
                "is disallowed, as no part of the sequence "
                "would be left to be used as current step."
                % (self.start_index, self.end_index)
            )

    def __len__(self):
        return (self.end_index - self.start_index + self.batch_size * self.stride) // (self.batch_size * self.stride)

    def __getitem__(self, index):
        rows = self.__index_to_row__(index)
        samples, y = self.__compile_batch__(rows)
        return samples, y

    def __index_to_row__(self, index): # 특정 배치 (index)를 구성할 row 목록을 반환한다. len(rows)는 batch size와 같다.
        if self.shuffle:
            rows = np.random.randint(self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size * self.stride, self.end_index + 1), self.stride)
        return rows

    def __compile_batch__(self, rows):  # 주어진 row 별 time series feature를 생성한다.
        samples = np.array([self.data[row - self.length: row: self.sampling_rate] for row in rows])
        if self.reverse:
            samples = samples[:, ::-1, ...]
        if self.length == 1:
            samples = np.squeeze(samples)

        if self.label is None:
            return samples, samples
        else:
            return samples, self.label[rows - self.length]

    @property
    def output_shape(self):
        x, y = self[0]
        return x.shape, y.shape

    @property
    def num_samples(self):
        count = 0
        for x, y in self:
            count += x.shape[0]
        return count

    def __str__(self):
        return '<TimeseriesGenerator data.shape={} / num_batches={:,} / output_shape={}>'.format(
            self.data.shape, len(self), self.output_shape,
        )

    def __repr__(self):
        return self.__str__()

class MergedTimeseriesGenerator(utils.Sequence):
    def __init__(self, data_list: list[TimeseriesGenerator], shuffle=False):
        super().__init__()
        self.data_list = data_list
        self.shuffle = shuffle
        self.index_map = []

        # 각 TimeseriesGenerator의 배치 인덱스를 전체 MergedTimeseriesGenerator의 인덱스로 매핑
        for d_idx, ds in enumerate(data_list):
            for i in range(len(ds)): # ds는 TimeseriesGenerator 인스턴스
                self.index_map.append((d_idx, i))

        #__str__를 위해 total_batches 계산
        self.total_batches = len(self.index_map)
        
        # shuffle=True일 경우, index_map을 섞음
        if self.shuffle:
            np.random.shuffle(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        if self.shuffle:
            actual_index = np.random.randint(0, len(self.index_map))
        else:
            actual_index = index
            
        d_idx, i = self.index_map[actual_index]
        x, y = self.data_list[d_idx][i]
        return x, y
    
    def on_epoch_end(self):
        pass


    @property
    def output_shape(self):
        # 첫 번째 제너레이터의 output_shape을 반환. 모든 제너레이터의 shape이 동일하다고 가정.
        if not self.data_list:
            return None, None
        x, y = self.data_list[0][0] # 첫 번째 제너레이터의 첫 번째 배치를 통해 shape 확인
        return x.shape[1:], y.shape[1:] # 배치 차원을 제외한 shape 반환
    
    def __str__(self):
        return '<MergedTimeseriesGenerator num_batches={:,} / output_shape={}>'.format(
            len(self), self.output_shape,
        )

    def __repr__(self):
        return self.__str__()

class AUCCheckCallback(callbacks.Callback):
    """
    Keras의 model.fit() 함수는 logs 딕셔너리에 있는 정보들을 에포크 끝에 표준 출력으로 보여주거나, TensorBoard 같은 로깅 도구에 전달.
    따라서 별도의 print() 문이 없어도, model.fit() 실행 시 에포크별 진행 상황과 함께 이 값들이 출력될 수 있음."""
    def __init__(self, model_mse: models.Model, tg_test: TimeseriesGenerator, gt):
        super().__init__()
        self.model_mse: models.Model = model_mse
        self.tg_test: TimeseriesGenerator = tg_test
        self.gt = gt # "ground truth"는 데이터의 실제 정답 레이블

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {} # logs가 None인 경우를 대비하여 딕셔너리로 초기화
        error = self.model_mse.predict(self.tg_test, verbose=0)

        # for auc-roc score
        auc_score = roc_auc_score(self.gt, error)
        logs['AUC'] = auc_score

        # # for f1-score
        # fpr, tpr, thresholds = roc_curve(self.gt, error) # (y_true, y_pred)
        # optimal_idx = (tpr - fpr).argmax() # 가로축이 fpr, 세로축이 tpr (fpr 변화 대비 tpr 변화가 커야해서)
        # optimal_threshold = thresholds[optimal_idx]
        # y_pred_optimal = (error >= optimal_threshold).astype(int) # binary 예측값 만들기
        # f1 = f1_score(self.gt, y_pred_optimal)
        # logs['f1-score'] = f1

        # # for precision
        # precision = precision_score(self.gt, y_pred_optimal)
        # logs['precision'] = precision

        # # for recall
        # recall = recall_score(self.gt, y_pred_optimal)
        # logs['recall'] = recall

        # y_true_for_plot = self.gt # 실제 레이블 (1이 intrusion)
        # y_probas_for_plot = np.column_stack([1 - error, error]) # Shape: (num_samples, 2) # # 정상 클래스(레이블 0)의 "확률"은 1 - error가 된다

        wandb.log({
            "epoch_auc": auc_score,
            "epoch": epoch,
            "train_loss": logs.get('loss'),
            "val_loss": logs.get('val_loss'),
            "mse_distribution": wandb.Histogram(error), # MSE 분포 시각화
        })  # wandb logging

        # wandb.log({
        #     "epoch_auc": auc_score, 
        #     "epoch_f1_score": f1,
        #     "epoch_precision": precision,
        #     "epoch_recall": recall,
        #     "epoch": epoch,
        #     "train_loss": logs.get('loss'),
        #     "val_loss": logs.get('val_loss'),
        #     "mse_distribution": wandb.Histogram(error), # MSE 분포 시각화
        #     "roc_curve": wandb.plot.roc_curve(
        #             y_true=y_true_for_plot,
        #             y_probas=y_probas_for_plot, # (num_samples, 2) 형태의 예측 확률
        #             labels=['Normal', 'Intrusion'], # 클래스 레이블
        #             classes_to_plot=[0, 1] # 정상(0)과 침입(1) 두 클래스 모두 플로팅
        #         )
        # })  # wandb logging

def main():
    # 입력받을 값 등록
    parser = argparse.ArgumentParser(description="Process some hyper parameter!")
    parser.add_argument('--batch_size', required=False, default=256, type=int)
    parser.add_argument('--epochs', required=False, default=2000, type=int)
    parser.add_argument('--window_size', required=False, default=150, type=int)
    parser.add_argument('--learning_rate', required=False, default=0.0001, type=float)
    parser.add_argument('--sampling_rate', required=False, default='0.005s', type=str)
    parser.add_argument('--hidden_space', required=False, default=125, type=int)

    # 입력받은 값을 args에 저장 (type: namespace)
    args = parser.parse_args()

    num_features = 107

    # 안전을 위해 argparse에서 받아온 값을 명시적으로 형 변환
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    window_size = int(args.window_size)
    lr = float(args.learning_rate)
    time = str(args.sampling_rate)
    hidden_space = int(args.hidden_space)

    print(f"==========이번 실험의 time sampling 기준: {time}==========")
    time_value = time[2:-1]  # '0.005'
    # WandB 초기화
    run = wandb.init(
        entity="alwaysy0ung-smwu",
        project="autoencoder-intrusion-detection",
        config={
            "architecture": "BiLSTM_Autoencoder",
            "dataset": "CAN_Bus_Intrusion",
            "window_size": window_size,
            "num_features": num_features,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate" : lr,
            "sampling_rate" : time,
            "hidden_space" : hidden_space,
            "optimizer": "adam",
            "loss": "mse"
        },
        name=f"autoencoder_run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    )

    print("--- 학습 데이터 로드 시작 (전처리된 parquet 파일) ---")
    import os
    current_directory = os.getcwd()
    print(current_directory)

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
    
    train = MergedTimeseriesGenerator([tg1, tg2, tg3, tg4], shuffle=True) # ,tg2, tg3, tg4

    print(f"raw 학습 데이터 길이: {df1.shape} + {df2.shape} + {df3.shape} + {df4.shape} -> 로드 완료: {len(train)}") # MergedTimeseriesGenerator가 제공할 수 있는 전체 배치의 총 개수(total number of batches). self.index_map의 길이와 같으며, 내부의 모든 TimeseriesGenerator들이 제공하는 배치들의 합계

    df_val_name = './cache/df5_merged_' + time_value + '.parquet'
    df_val = pd.read_parquet(df_val_name)
    validation = TimeseriesGenerator(data=df_val.to_numpy(), length=window_size, shuffle=True)

    print(f"검증 데이터 길이: {df1.shape} -> 로드 완료: {len(validation)}")  

    df_test_one_name = './cache/fabrication/test_fabrication_aid=316_' + time_value + '.parquet'
    df_test = pd.read_parquet(df_test_one_name) # 최종 모델에 대해 모든 test file 성능 측정 전, 에포크마다 추적할 하나의 대표 테스트셋을 선정. ('test_results_for_table'를 추가하기 이전처럼 단일 테스트셋에 대한 AUC, F1-score 등을 로깅)
    df_test_label = df_test[['label']].copy() # Time 컬럼이 인덱스 컬럼이라 다음과 같이 할 수 없음:     df_test_label = df_test[['Time', 'label']].copy()
    df_test = df_test.drop('label', axis=1)
    gt_ = df_test_label.rolling(window_size).max().dropna() # preprocessing 단계에 없고 여기서 처리해주어야함
    assert df_test.shape[0] - window_size + 1 == gt_.shape[0], '제~~~~~~~~~~발 이러지 말어요'
    test = TimeseriesGenerator(data=df_test.to_numpy(), length=window_size, label = gt_)

    # 모델 선택하기
    model = BiLSTM(window_size=window_size, batch_size=batch_size, hidden_space = hidden_space)
    # 학습률(learning rate) 설정 및 compile
    custom_adam = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=custom_adam, loss='mse')

    model_mse = make_model_mse(model)
    model_mse.compile(loss='mse') #make_model_mse는 학습 가능한 가중치를 가지지 않기 떄문에 optimizer를 설정할 필요가 없음

    model_mse.summary()

    acb = AUCCheckCallback(model_mse, test, gt=gt_)

    # wandb_cb = WandbCallback()

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', # 또는 'AUC'로 설정하고 mode='max'로 변경
        patience=50,        # N 에포크 동안 개선 없으면 중단 (2000 에포크에 50값)
        restore_best_weights=True # 가장 좋은 가중치 복원
    )

    name = f'autoencoder_model_{pd.Timestamp.now().strftime("%y%m%d_%H%M")}.keras' # 학습 시작 시점에 이름 정의
    try:
        history = model.fit(
            train,
            epochs=epochs,
            validation_data=validation,
            callbacks=[acb, early_stopping] # acb 인스턴스에 정의된 콜백 기능들을 모델 학습 과정 중에 활성화하겠다 # wandb_cb도 있었는데 삭제함
        )
    except KeyboardInterrupt:
        print("\n학습이 사용자 요청에 의해 중단되었습니다. 현재까지의 모델로 평가를 진행합니다.")
        # EarlyStopping의 restore_best_weights=True가 설정되어 있다면, 이 시점에서 모델에는 이미 best weights가 로드되어 있을 것. 따라서 별도로 model.load_weights()를 호출할 필요가 없음.
        pass # 예외 발생 시에도 코드의 다음 부분으로 넘어감
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        sys.exit(1) # 프로그램 종료

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

                print(f"[test_data_info]\ntest_shape: {df_test.shape}\nnum_anomalies: {int(gt_.sum().iloc[0])}\nanomaly_ratio: {float(gt_.mean().iloc[0])}")

                # test auc
                y_pred = model_mse.predict(test, verbose=1) # 여기서는 model_mse가 '오차'를 출력하도록 설계되었으므로, 예측 결과가 바로 오차다 (y_pred)
                y_true = gt_.values # (pandas Series 객체) -> (NumPy 배열) # (y_true)
                test_auc = roc_auc_score(y_true, y_pred)

                fpr, tpr, thresholds = roc_curve(y_true, y_pred) # (y_true, y_pred)
                optimal_idx = (tpr - fpr).argmax() # 가로축이 fpr, 세로축이 tpr (fpr 변화 대비 tpr 변화가 커야해서)
                optimal_threshold = thresholds[optimal_idx]
                y_pred_optimal = (y_pred >= optimal_threshold).astype(int) # binary 예측값 만들기

                test_f1 = f1_score(y_true, y_pred_optimal)
                test_precision = precision_score(y_true, y_pred_optimal)
                test_recall = recall_score(y_true, y_pred_optimal)

                print(f"    - AUC: {test_auc:.4f}, F1: {test_f1:.4f}, P: {test_precision:.4f}, R: {test_recall:.4f}, optimal_threshold: {optimal_threshold}")
                _, sub_identifier = test_pp.get_attack_type_from_filename(test_file.name) # : test_file은 경로를 포함하므로
                test_results_for_table.append({
                    "Attack_Type": attack_type,
                    "Info": sub_identifier,
                    "AUC": test_auc,
                    "F1_Score": test_f1,
                    "Precision": test_precision,
                    "Recall": test_recall
                })
                flag+=1

    df_test_performance = pd.DataFrame(test_results_for_table) # 딕셔너리 리스트를 Pandas DataFrame으로 변환
    test_performance_table = wandb.Table(dataframe=df_test_performance) # Pandas DataFrame을 wandb.Table의 data 인자로 전달
    wandb.log({"final_test_performance_table": test_performance_table})

    wandb.log({"time resample frequency": time})
    print(f"\n{len(test_results_for_table)} 길이의 최종 test 결과가 WandB Table에 logged됨.")

    # ===== 모델 저장 =====
    models_dir = Path('./models')
    model_path = models_dir / name
    model.save(model_path)
    print(f"\n[models]: '{name}' has been saved in '{model_path}'") # 저장할 건 model_mse가 아닌 conv1D. 왜냐면 model_mse는 새로운 가중치를 학습하지 않음. 단순히 conv1D의 입출력 텐서를 사용하여 K.mean(K.square(input_s - output_s), axis=[1, 2]) 연산을 수행하는 연산 그래프 또는 함수 역할이므로.

    # 모델을 wandb artifact로 저장
    model_artifact = wandb.Artifact(
        name="autoencoder_model",
        type="model",
        description="BiLSTM Autoencoder for CAN bus intrusion detection"
    )
    model_artifact.add_file(str(model_path)) # model_path는 path객체이므로 str 형변환이 필요
    run.log_artifact(model_artifact)

    # 학습 이력 요약
    wandb.run.summary["total_epochs"] = epochs
    wandb.run.summary["model_saved"] = name

    # wandb 세션 종료
    run.finish()
    print("WandB 로깅 완료.")

if __name__ == "__main__":
    main()