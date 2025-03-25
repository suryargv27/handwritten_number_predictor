import keras
from keras.models import load_model
from keras.datasets import mnist
model = load_model('mnist.h5')

num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_test = x_test.astype('float32')
x_test /= 255


score = model.evaluate(x_test, y_test, verbose=0)
print(model.summary())
print()
print('Test loss :', score[0])
print('Test accuracy :', score[1])

