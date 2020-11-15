import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *
from keras import layers
import random
import tensorflow as tf
traintitles=pd.read_csv("train/ImageSets/Segmentation/train.txt",header=None,dtype=str,names=["traintitles"])
segarr=[]
imarr=[]

def crop(im,segim):
    w=random.randint(0,len(im[0])-224)
    h=random.randint(0,len(im)-224)
    return [im[h:h+224,w:w+224],segim[h:h+224,w:w+224]]

c=0
for i in traintitles["traintitles"]:
    segim=plt.imread("train/SegmentationClass/"+i+".png")
    im=plt.imread("train/JPEGImages/"+i+".jpg")
    if len(segim)>=224 and len(segim[0])>=224 and len(im)>=224 and len(im[0])>=224:
        c=c+1
        arr=crop(im,segim)
        segarr.append(arr[1])
        imarr.append(arr[0])
print(len(segarr))


def types(segarr):
    pixeltypes=[]
    for im in segarr:
        for j in range(len(im)):
            for k in range(len(im[0])):
                if [im[j][k][0],im[j][k][1],im[j][k][2]] not in pixeltypes:
                    pixeltypes.append([im[j][k][0],im[j][k][1],im[j][k][2]])
        if len(pixeltypes)==22:
            break
    return np.array(pixeltypes)

def getitem(segarr,pixeltypes,count,numb):
    arr22=[]
    for i in range(count):
        im=segarr[i]
        im22=np.zeros((len(im),len(im[0]),numb))
        print(count-i)
        for c in range(numb):
            for j in range(len(im)):
                for k in range(len(im[0])):
                    if im[j][k][0]==pixeltypes[c][0] and im[j][k][1]==pixeltypes[c][1] and im[j][k][2]==pixeltypes[c][2]:
                        im22[j][k][c]=1;
        arr22.append(im22)
    return np.array(arr22)
def topict(im22):
    im=np.zeros((len(im22),len(im22[0]),3))
    for j in range(len(im)):
        for k in range(len(im[0])):
            cons=0
            for c in range(22):
                if im22[j][k][c]>=cons:
                    im[j][k][0]=pixeltypes[c][0]
                    im[j][k][1]=pixeltypes[c][1]
                    im[j][k][2]=pixeltypes[c][2]
                    cons=im22[j][k][c]
    return im







pixeltypes=types(segarr)
arr22=getitem(segarr,pixeltypes,10,22)
imarr=np.array(imarr)


model=Sequential(keras.applications.VGG16(include_top=False,weights="imagenet",input_tensor=None,input_shape=None,pooling=None,classes=1000,classifier_activation="softmax",))
model.add(layers.UpSampling2D(2))
model.add(layers.Conv2D(filters=128,kernel_size=3,padding="same"))
model.add(layers.UpSampling2D(2))
model.add(layers.Conv2D(filters=64,kernel_size=3,padding="same"))
model.add(layers.UpSampling2D(2))
model.add(layers.Conv2D(filters=22,kernel_size=3,padding="same"))
model.add(layers.UpSampling2D(2))
model.add(layers.Conv2D(filters=22,kernel_size=3,padding="same"))
model.add(layers.UpSampling2D(2))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')


model.fit(imarr[:10],arr22[:10],epochs=1,batch_size=10)

a=model.predict(imarr[0:10])
for i in range(1):
    b=topict(a[i])
    plt.imshow(b)
    plt.show()
    plt.imshow(imarr[i])
    plt.show()
    plt.imshow(segarr[i])
    plt.show()