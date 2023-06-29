#Import section.
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

#Loading datasets.
path='PetImages/'
categories=['cat', 'dog']

img_list=[]
img_labels=[]
for category in categories:
    dir=os.path.join(path, category)
    directory=os.listdir(dir)
    label= 1 if(category=='cat') else 0
    for image in directory:
        impath=os.path.join(dir,image)
        picture=cv2.resize(cv2.imread(impath, cv2.IMREAD_GRAYSCALE), (100, 100))
        img_list.append(picture)
        img_labels.append(label)

img_list=np.array(img_list)/255.0
img_labels=np.array(img_labels)

#Splitting dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(img_list, img_labels, random_state=42, shuffle=True, test_size=.2)
print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)
X_train = X_train.reshape(-1, 100, 100, 1)

#CNN Model.
model = Sequential()
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, input_shape = X_train.shape[1:], activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Model Training and Evaluation.
model.fit(X_train, Y_train, epochs=15, batch_size=64)
model.evaluate(X_test, Y_test)

#MAKING PREDICTIONS.

idx2 = random.randint(0, len(Y_test)-1)
plt.imshow(X_test[idx2, :])
plt.show()

y_pred = model.predict(X_test[idx2, :].reshape(-1, 100, 100, 1))
print(y_pred)
if (y_pred[0][0]>y_pred[0][1]):
    pred = 'dog'
else:
    pred = 'cat'

print("Our model says it is a :", pred)

#END OF CODE.