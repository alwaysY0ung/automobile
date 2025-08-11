from model_BiLSTM import BiLSTM

window_size_value = 300

model1 = BiLSTM()
model1.summary(expand_nested=True)

model2 = BiLSTM(window_size=200)
model2.summary(expand_nested=True)

model3 = BiLSTM(window_size=window_size_value)
model3.summary(expand_nested=True)