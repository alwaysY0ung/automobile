from keras import layers
from keras import models
import keras.api.ops as K

def make_conv1D(window_size = 150, batch_size = 64):
    # encoder
    h_dim = 8
    
    # Input layer
    num_features = 107
    input_shape = (window_size, num_features)

    # Encoder (150 -> 128 -> 64 -> 32 -> 16 -> 8)
    x = input_layer = encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same')(x)  # (None, 150, 32)
    x = layers.Cropping1D(cropping=(11, 11))(x)  # (None, 128, 32)
    x = layers.Conv1D(filters=16, kernel_size=5, padding='same')(x)  # (None, 128, 16)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)  # (None, 64, 16)
    x = layers.Conv1D(filters=8, kernel_size=5, padding='same')(x)  # (None, 64, 8)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)  # (None, 32, 8)
    x = layers.Conv1D(filters=4, kernel_size=5, padding='same')(x)  # (None, 32, 4)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)  # (None, 16, 4)
    x = layers.Conv1D(filters=2, kernel_size=5, padding='same')(x)  # (None, 16, 2)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)  # (None, 8, 2)
    x = layers.Flatten()(x)  # (None, 16)
    encoder_output = layers.Dense(h_dim)(x)
    encoder = models.Model(inputs=encoder_input, outputs=encoder_output, name="encoder_model")

    # Decoder (8 -> 16 -> 32 -> 64 -> 128 -> 150)
    decoder_input = layers.Input(shape=(h_dim,))
    x = layers.Dense(16)(decoder_input) # (None, 16)
    x = layers.Reshape((8, 2))(x)  # (None, 8, 2)
    x = layers.UpSampling1D(size=2)(x)  # (None, 16, 2)
    x = layers.Conv1D(filters=4, kernel_size=5, padding='same')(x)  # (None, 16, 4)
    x = layers.UpSampling1D(size=2)(x)  # (None, 32, 4)
    x = layers.Conv1D(filters=8, kernel_size=5, padding='same')(x)  # (None, 32, 8)
    x = layers.UpSampling1D(size=2)(x)  # (None, 64, 8)
    x = layers.Conv1D(filters=16, kernel_size=5, padding='same')(x)  # (None, 64, 16)
    x = layers.UpSampling1D(size=2)(x)  # (None, 128, 16)
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same')(x)  # (None, 128, 32)
    x = layers.ZeroPadding1D(padding=(11, 11))(x)  # (None, 150, 32)
    decoder = layers.Conv1D(filters=1, kernel_size=5, activation='linear', padding='same')(x)  # (None, 150, 1)
    decoder_output = layers.Conv1D(filters=num_features, kernel_size=5, activation='linear', padding='same')(x) # 필터 수를 num_features로 변경
    decoder = models.Model(inputs=decoder_input, outputs=decoder_output, name="decoder_model")

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