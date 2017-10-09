import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
data_dir = ["none","fist","hand","one","peace"]

allX=None
allY=None
for i in range(len(data_dir)):
    path = "data/"+data_dir[i]+"/"
    data_list = os.listdir(path)
    X = np.empty((len(data_list),1,50,50))
    y = np.ones(len(data_list))*i 
    for num in range(len(data_list)):
        X[num,0,:,:] = plt.imread(path+data_list[num])
    
    if allX is None:
        allX=X
        allY=y
    else:
        allX = np.concatenate((allX,X),0)
        allY = np.concatenate((allY,y),0)

s = range(allX.shape[0])
np.random.shuffle(s)
allX = allX[s]
allY = allY[s]

mean_px = np.mean(allX)
std_px = np.std(allX)
#print(mean_im.shape)
allX = (allX - mean_px)/std_px

print(allY)

trainX = allX[:3000]
trainY = allY[:3000]

validationX = allX[3000:3500]
validationY = allY[3000:3500]

testX = allX[3500:]
testY = allY[3500:]

data = {
    "trainX":trainX.tolist(),
    "trainY":trainY.tolist(),
    "validX":validationX.tolist(),
    "validY":validationY.tolist(),
    "testX":testX.tolist(),
    "testY":testY.tolist()
}

json.dump(data,open("data/data.json","wb"))
