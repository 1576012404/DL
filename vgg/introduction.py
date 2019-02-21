import pickle
import numpy as np
import os
import matplotlib.pyplot  as plt
from matplotlib.pyplot import imshow

CIFAR_DIR="./cifar-10-batches-py"

print(os.listdir(CIFAR_DIR))
with open(os.path.join(CIFAR_DIR,"data_batch_1"),"rb") as f:
    data=pickle.load(f,encoding='bytes')
    print("all",type(data),data.keys())
    print("data",data[b"data"].shape )
    print("labels",data[b"labels"][0:2])

image_arr=data[b"data"][102]
image_arr=image_arr.reshape((3,32,32))
image_arr=image_arr.transpose((1,2,0))

imshow(image_arr)
plt.show()
