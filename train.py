import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

path = "myData"
testRatio = 0.2
valRatio = 0.2

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
        curentImg = cv2.resize(curentImg, (32, 32))
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

plt.figure(figsize=(10,5))
plt.bar(range(0,numOfClasses), numOfSamples)
plt.title("Number of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


