import matplotlib.pyplot as plt

print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Utilis import *
from sklearn.model_selection import train_test_split



#step 1
path = 'myData'
data = importDataInfo(path)

#step 2 vizualization and distribution of data
data=balanceData(data,display=False)

#step 3 procesing
imagesPath,steerings = loadData(path,data)
#print(imagesPath[0],steering[0])

#step 4 spliting of data into traning data and validation data
xTrain,xVal,yTrain,yVal = train_test_split(imagesPath,steerings,test_size=0.2,random_state=5)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#step 5 agumentation of data -add more variety and variance which helps in trainning process to generalize more efficiently

#step 6 preprocesing

#step 7 batch generator

#step 8 create model - complie model
model=creatModel()
model.summary()

#step 9 tranning model
history=model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)

#step 10
model.save('model.h5')
print('model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()




