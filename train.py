import numpy as np
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

path = "myData"
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32, 32, 3)
batchSizeVal = 50
epochsVal = 10
stepsPerEpoch = 2000

images = []
classNum = []

myList = os.listdir(path)

numOfClasses = len(myList)
print("Total No of Classes Detected", numOfClasses)

print("Importing Classes...")
for x in range(0, numOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curentImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curentImg = cv2.resize(curentImg, (imageDimensions[0], imageDimensions[1]))
        images.append(curentImg)
        classNum.append(x)
    print(x, end=" ")
print(" ")

images = np.array(images)
classNum = np.array(classNum)
print(images.shape)

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(images, classNum, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for x in range(0, numOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))
print(numOfSamples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, numOfClasses), numOfSamples)
plt.title("Number of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_validation = np.array(list(map(preprocessing, X_validation)))

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1))

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
y_train = to_categorical(y_train, numOfClasses)
y_test = to_categorical(y_test, numOfClasses)
y_validation = to_categorical(y_validation, numOfClasses)


def myModel():
    numOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    numOfNodes = 500

    model = Sequential()
    model.add((Conv2D(numOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                                imageDimensions[1],
                                                                1), activation='relu')))
    model.add((Conv2D(numOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(numOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(numOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(numOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batchSizeVal), steps_per_epoch=stepsPerEpoch,
                              epochs=epochsVal, validation_data=(X_validation, y_validation),
                              shuffle=1)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title("Loss")
plt.xlabel("Epochs")

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print("Test Score = ", score[0])
print("Test Accuracy = ", score[1])

pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
