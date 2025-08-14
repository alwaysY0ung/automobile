from keras import callbacks
from keras import layers
from keras import models
from keras import optimizers
from keras import utils
import keras.api.ops as K

def make_model_mse(model):
    input_s = model.inputs[0]
    output_s = model.outputs[0]
    error_s = K.mean(K.square(input_s - output_s), axis=[1, 2])
    return models.Model(inputs=input_s, outputs=error_s)

def make_model_mse2(model):
    input_s = model.inputs[0]
    output_s = model.outputs[0]
    error_s = K.mean(K.square(input_s - output_s), axis=[1, ])
    return models.Model(inputs=input_s, outputs=error_s)

import subprocess
def gpu_memory_info():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, text=True
    )
    total, used, free = map(int, result.stdout.strip().split('\n')[0].split(','))
    print(f"GPU Total: {total} MB")
    print(f"GPU Used : {used} MB")
    print(f"GPU Free : {free} MB")

def model_logging(model):
    pass

class NotScaledError(Exception):
    def __init__(self, message="데이터가 0~1 범위를 벗어났습니다"):
        super().__init__(message)
        self.message = message

def ScaleChecker(df):
    bounds = df.agg(['min', 'max'])
    in_range = (bounds.loc['min'] >= 0) & (bounds.loc['max'] <= 1)
    is_scaled = in_range.all()
    if not is_scaled:
        raise NotScaledError
    return True


if __name__ == "__main__":
    model = model_mse2()