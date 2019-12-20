from keras.models import load_model

model = load_model('MNIST-CNN.model')

model.summary()