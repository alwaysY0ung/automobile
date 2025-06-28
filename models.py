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

def make_conv1D(window_size = 150, batch_size = 64):
    num_features = 107

    input_shape = (window_size, num_features)

    # encoder
    h_dim = 8

    # inputs_e = layers.Input(shape=input_shape)
    # hidden1_e = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs_e)
    # hidden2_e = layers.MaxPooling1D(pool_size=3, padding='same')(hidden1_e)
    # hidden3_e = layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(hidden2_e)
    # hidden4_e = layers.MaxPooling1D(pool_size=5, padding='same')(hidden3_e)
    # hidden5_e = layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding='same')(hidden4_e)
    # flatten_e = layers.Flatten()(hidden5_e)
    # outputs_e = layers.Dense(h_dim, activation='relu')(flatten_e)
    # # outputs_e = out_1 = layers.Dense(h_dim, activation='relu')(flatten_e)

    x = input_layer = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=3, padding='same')(x)
    x = layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=5, padding='same')(x)
    x = layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding='same')(x) # padding='same'으로 인해 window_size (10)는 유지되고 필터 수가 8로 바뀜
    # x.shape는 (None, timesteps, features) 형태이므로, [1:]을 사용하여 (timesteps, features)
    flattened_pre_shape = x.shape[1:] # (10, 8)

    x = layers.Flatten()(x)
    x = output_layer = layers.Dense(h_dim, activation='relu')(x)

    """
    병렬연산
    """
    # hidden1_e = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs_e)
    # hidden2_e = layers.MaxPooling1D(pool_size=3, padding='same')(hidden1_e)
    # hidden3_e = layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(hidden2_e)
    # hidden4_e = layers.MaxPooling1D(pool_size=5, padding='same')(hidden3_e)
    # hidden5_e = layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding='same')(hidden4_e)
    # flatten_e = layers.Flatten()(hidden5_e)
    # outputs_e = out_2 = layers.Dense(h_dim, activation='relu')(flatten_e)

    # out = out_1 + out_2

    encoder = models.Model(inputs=input_layer, outputs=output_layer, name="encoder_model") # 병렬연산: outputs=out

    # decoder
    inputs_d = layers.Input(shape=(h_dim,))
    # Flatten 직전의 shape을 사용
    reshaped_timesteps = flattened_pre_shape[0] # 10
    reshaped_features = flattened_pre_shape[1] # 8

    x = layers.Dense(reshaped_timesteps * reshaped_features, activation='relu')(inputs_d)
    x = layers.Reshape((reshaped_timesteps, reshaped_features))(x)

    x = layers.UpSampling1D(size=5)(x)
    x = layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)

    x = layers.UpSampling1D(size=3)(x)
    x = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)

    outputs_d = layers.Conv1D(filters=num_features, kernel_size=5, activation='sigmoid', padding='same')(x)

    decoder = models.Model(inputs=inputs_d, outputs=outputs_d, name="decoder_model")

    """
    인코더 (Encoder)
    Input: (None, 150, 107) (window_size, num_features)
    layers.Conv1D(filters=32, kernel_size=5, padding='same'): (None, 150, 32)
    layers.MaxPooling1D(pool_size=2, padding='same'): (None, 75, 32)
    layers.Conv1D(filters=16, kernel_size=5, padding='same'): (None, 75, 16)
    layers.MaxPooling1D(pool_size=2, padding='same'): (None, 38, 16)
    layers.Conv1D(filters=8, kernel_size=5, padding='same'): (None, 38, 8)
    layers.Flatten(): (None, 38 * 8) (즉, (None, 304))
    layers.Dense(h_dim=8): (None, 8)

    디코더 (Decoder)
    Input: (None, 8) (잠재 공간 h_dim)
    layers.Dense(reshaped_timesteps * reshaped_features): (None, 38 * 8) (즉, (None, 304))
    layers.Reshape((reshaped_timesteps, reshaped_features)): (None, 38, 8)
    layers.UpSampling1D(size=2): (None, 76, 8)
    layers.Conv1D(filters=16, kernel_size=5, padding='same'): (None, 76, 16)
    layers.UpSampling1D(size=2): (None, 152, 16)
    layers.Conv1D(filters=32, kernel_size=5, padding='same'): (None, 152, 32)
    layers.Cropping1D(cropping=(1, 1)): (None, 150, 32)
    layers.Conv1D(filters=num_features=107, kernel_size=5, activation='sigmoid', padding='same'): (None, 150, 107) (최종 출력)
    """
    # Autoencoder 전체 모델
    autoencoder_input = layers.Input(shape=input_shape)
    autoencoder_encoded = encoder(autoencoder_input)
    autoencoder_decoded = decoder(autoencoder_encoded)

    autoencoder = models.Model(inputs=autoencoder_input, outputs=autoencoder_decoded, name="autoencoder_model")

    # autoencoder.compile(
    #     optimizer=optimizers.Adam(learning_rate=1e-4),
    #     loss='mse'
    # )

    return autoencoder

def make_model_mse(model):
    input_s = model.inputs[0]
    output_s = model.outputs[0]
    error_s = K.mean(K.square(input_s - output_s), axis=[1, 2])
    return models.Model(inputs=input_s, outputs=error_s)

if __name__ == '__main__':
    model = make_conv1D()
    model.summary(expand_nested=True)