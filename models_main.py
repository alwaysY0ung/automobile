import numpy as np
from keras import callbacks
from keras import layers
from keras import models
from keras import optimizers
from keras import utils
from sklearn.metrics import roc_auc_score
from pathlib import Path
import pandas as pd
import keras.api.ops as K
import cantools
from models import make_conv1D, make_model_mse
# from data_loader import DataLoader # 캐시 관리
# from raw_data_processor import RawDataParser # step1 관련
# from data_decoder import DataDecoder # step2 관련

# from preproccessing import Preprocessor

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
    def __init__(self, model_mse: models.Model, tg_test: TimeseriesGenerator, gt):
        super().__init__()
        self.model_mse: models.Model = model_mse
        self.tg_test: TimeseriesGenerator = tg_test
        self.gt = gt # "ground truth"는 데이터의 실제 정답 레이블

    def on_epoch_end(self, epoch, logs=None):
        error = self.model_mse.predict(self.tg_test, verbose=0)
        auc_score = roc_auc_score(self.gt, error)
        logs['AUC'] = auc_score

# --- make_model_mse 함수 정의 ---
def make_model_mse(model):
    input_s = model.inputs[0]
    output_s = model.outputs[0]
    error_s = K.mean(K.square(input_s - output_s), axis=[1, 2])
    return models.Model(inputs=input_s, outputs=error_s)

def main():
    global num_features
    window_size = 150
    batch_size = 64
    num_features = 107

    print("--- 학습 데이터 로드 시작 (전처리된 parquet 파일) ---")
    import os
    current_directory = os.getcwd()
    print(current_directory)
    
    df1 = pd.read_parquet('./cache/df1_merged_0025.parquet')
    df2 = pd.read_parquet('./cache/df2_merged_0025.parquet')
    df3 = pd.read_parquet('./cache/df3_merged_0025.parquet')
    df4 = pd.read_parquet('./cache/df4_merged_0025.parquet')

    tg1 = TimeseriesGenerator(data=df1.to_numpy(), length=window_size, shuffle=True)
    tg2 = TimeseriesGenerator(data=df2.to_numpy(), length=window_size, shuffle=True)
    tg3 = TimeseriesGenerator(data=df3.to_numpy(), length=window_size, shuffle=True)
    tg4 = TimeseriesGenerator(data=df4.to_numpy(), length=window_size, shuffle=True)
    
    train = MergedTimeseriesGenerator([tg1,tg2, tg3, tg4], shuffle=True)

    print(f"학습 데이터 로드 완료.")

    df_val = pd.read_parquet('./cache/df5_merged_0025.parquet')
    validation = TimeseriesGenerator(data=df_val.to_numpy(), length=window_size, shuffle=True)

    print(f"검증 데이터 로드 완료.")  

    # test_parquet_files = sorted(list(Path('./datasets/intrusion_datasets').glob('*.parquet')))
    test_parquet_files = sorted(list(Path('./cache/fuzz').glob('*.parquet')))
    df_test = pd.read_parquet(test_parquet_files[0])
    df_test_label = df_test[['label']].copy() # Time 컬럼이 인덱스 컬럼이라 다음과 같이 할 수 없음:     df_test_label = df_test[['Time', 'label']].copy()
    df_test = df_test.drop('label', axis=1)
    gt_ = df_test_label.rolling(window_size).max().dropna() # 난 이 단계가 이미 preprocessing에서 된거 아닌가? 아 아니네
    # print(df_test.shape[0] - window_size + 1)
    # print(gt_.shape[0])
    assert df_test.shape[0] - window_size + 1 == gt_.shape[0], '제~~~~~~~~~~발 이러지 말어요'
    test = TimeseriesGenerator(data=df_test.to_numpy(), length=window_size, label = gt_)


    conv1D = make_conv1D()
    conv1D.compile(optimizer='adam', loss='mse')

    model_mse = make_model_mse(conv1D)
    model_mse.compile(optimizer='adam', loss='mse')

    model_mse.summary()

    acb = AUCCheckCallback(model_mse, test, gt=gt_)

    conv1D.fit(
        train,
        epochs=10,
        validation_data=validation,
        callbacks=[acb] # acb 인스턴스에 정의된 콜백 기능들을 모델 학습 과정 중에 활성화하겠다
    )

    # train 및 validation 완료

    # AUC 점수 계산
    y_pred = model_mse.predict(test, verbose=1) # 여기서는 model_mse가 '오차'를 출력하도록 설계되었으므로, 예측 결과가 바로 오차다 (y_pred)
    y_true = gt_.values # (pandas Series 객체) -> (NumPy 배열) # (y_true)
    final_test_auc = roc_auc_score(y_true, y_pred)
    print(f"\n최종 테스트 데이터 AUC: {final_test_auc:.6f}")

    # --- 추가: 모델 저장 ---
    name = 'autoencoder_model_250627_1400.keras'
    conv1D.save(name)
    print(f"\n학습된 오토인코더 모델이 '{name}'로 저장되었습니다.") # 저장할 건 model_mse가 아닌 conv1D. 왜냐면 model_mse는 새로운 가중치를 학습하지 않음. 단순히 conv1D의 입출력 텐서를 사용하여 K.mean(K.square(input_s - output_s), axis=[1, 2]) 연산을 수행하는 연산 그래프 또는 함수 역할이므로.


if __name__ == "__main__":
    main()