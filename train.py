import numpy as np
import cv2
import os

path = "myData"

images = []
classNum = []
myList = os.listdir(path)
print(myList)
numOfClasses = len(myList)

for x in range(0, numOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curentImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curentImg = cv2.resize(curentImg, (32,32))
        images.append(curentImg)
        classNum.append(x)
    print(x)

