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