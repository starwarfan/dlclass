from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import sklearn.datasets
import datetime
import matplotlib.pyplot as plt
import numpy as np

train_libsvm = "../data/mnist"
test_libsvm = "../data/mnist.t"

X_train, y_train = sklearn.datasets.load_svmlight_file(train_libsvm, n_features=784)
X_train = X_train.toarray()
X_test, y_test = sklearn.datasets.load_svmlight_file(test_libsvm, n_features=784)
X_test = X_test.toarray()

model = Sequential()
model.add(Dense(200, input_dim=784, kernel_initializer='normal', activation='relu'))
#model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

categorical_labels = to_categorical(y_train, num_classes=10)
start = datetime.datetime.today()

history = model.fit(X_train, categorical_labels, epochs=100, batch_size=100)

scores = model.evaluate(X_test, to_categorical(y_test, num_classes=10))

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
