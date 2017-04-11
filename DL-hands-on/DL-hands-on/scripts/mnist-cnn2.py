from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import sklearn.datasets
import datetime
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

train_libsvm = "../data/mnist"
test_libsvm = "../data/mnist.t"

X_train, y_train = sklearn.datasets.load_svmlight_file(train_libsvm, n_features=784)
X_train = X_train.toarray()
X_train = X_train.reshape( (X_train.shape[0], 28, 28, 1) )
X_train = X_train.astype('float32')
X_train /= 255
y_train = to_categorical(y_train, num_classes=10)

X_test, y_test = sklearn.datasets.load_svmlight_file(test_libsvm, n_features=784)
X_test = X_test.toarray()
X_test = X_test.reshape( (X_test.shape[0], 28, 28, 1) )
X_test = X_test.astype('float32')
X_test /= 255
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(Conv2D(8, # number of kernels 
						(4, 4), # kernel size
                        padding='valid',
                        input_shape=(28, 28, 1)))

model.add(Activation('relu'))

model.add(Conv2D(8, (4, 4)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = datetime.datetime.today()

history = model.fit(X_train, y_train, batch_size=128, epochs=15, verbose=1)

scores = model.evaluate(X_test, y_test, verbose=1)

print
for i in range(len(model.metrics_names)):
	print("%s: %f" % (model.metrics_names[i], scores[i]))

print ("Start: " + str(start))
end = datetime.datetime.today()
print ("End: " + str(end))
print ("Elapse: " + str(end-start))

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
