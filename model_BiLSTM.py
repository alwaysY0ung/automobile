from keras import layers
from keras import models
from model_manager import make_model_mse2

def BiLSTM(window_size = 150, batch_size = 64, hidden_space = 125):
    # Input layer
    num_features = 107 # Input features (107)
    input_shape = (window_size, num_features) # (150, 107)

    x = input_layer = layers.Input(shape = input_shape) # (150, 107)
    x = layers.Bidirectional(layers.LSTM(units=107, return_sequences=True))(x) # BiLSTM 1 (Output shape: 150, 214) # , dropout=0.3

    x = layers.Bidirectional(layers.LSTM(units=hidden_space))(x) # BiLSTM 2 (Output shape: 250)

    x = layers.RepeatVector(window_size)(x) # Repeat Vector (Output shape: 150, 250)
    x = layers.Bidirectional(layers.LSTM(units=107, return_sequences=True))(x) # BiLSTM 3 (Output shape: 150, 214)

    x = layers.Bidirectional(layers.LSTM(units=107, return_sequences=True))(x) # BiLSTM 4 (Output shape: 150, 214)

    x = layers.TimeDistributed(layers.Dense(num_features, activation=None))(x) # Time-distributed dense layer (Output shape: 150, 107) # S와 S'를 비교해야하는 오토인코더 컨셉을 생각했을 때 출력값의 범위를 인위적으로 제한하게 되는 activation function은 쓰지 않는 게 좋을 듯하다.

    model = models.Model(inputs=input_layer, outputs=x, name="autoencoder_model")

    return model

if __name__ == '__main__':
    model = make_model_mse2(BiLSTM())
    model.summary(expand_nested=True)