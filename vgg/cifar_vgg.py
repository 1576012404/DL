import tensorflow as tf
import numpy as np


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle

CIFAR_DIR="./cifar-10-batches-py"

print(os.listdir(CIFAR_DIR))



def load_data(filename):
    with open(filename,"rb") as f:
        data=pickle.load(f,encoding='bytes')
        return data[b"data"],data[b"labels"]

class CifarData():
    def __init__(self,filenames,need_shuffle):
        all_data=[]
        all_labels=[]
        for filename in filenames:
            data,labels=load_data(filename)
            all_data.append(data)
            all_labels.append(labels)

        self._data=np.vstack(all_data)
        self._data=self._data/127.5-1
        self._labels=np.hstack(all_labels)
        print("shape",self._data.shape,self._labels.shape)
        self._num_examples=self._data.shape[0]
        print("self.num_exampels",self._num_examples)
        self._need_shuffle=need_shuffle
        self._indicator=0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        p=np.random.permutation(self._num_examples)
        self._data=self._data[p]
        self._labels=self._labels[p]

    def next_batch(self,batch_size):
        end_indicator=self._indicator+batch_size
        if end_indicator>self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator=0
                end_indicator=batch_size
            else:
                raise Exception("have no more example")
        if end_indicator>self._num_examples:
            raise Exception("batch size too large")
        batch_data=self._data[self._indicator:end_indicator]
        batch_labels=self._labels[self._indicator:end_indicator]
        self._indicator=end_indicator
        return batch_data,batch_labels



train_filenams=[os.path.join(CIFAR_DIR,"data_batch_%d"%i) for i in range(1,6)]
test_filenames=[os.path.join(CIFAR_DIR,"test_batch")]

train_data=CifarData(train_filenams,True)






x=tf.placeholder(tf.float32,[None,3072])
y=tf.placeholder(tf.int64,[None])

x_image=tf.reshape(x,[-1,3,32,32])
x_image=tf.transepose(x_image,perm=[0,2,3,1])

conv1_1=tf.layers.conv2d(x_image,32,(3,3),padding="same",activation=tf.nn.relu,name="conv1_1")
conv1_2=tf.layers.conv2d(conv1_1,32,(3,3),padding="same",activation=tf.nn.relu,name="conv1_2")
pooling1=tf.layers.max_pooling2d(conv1_2,(2,2),(2,2),name="pool1")

conv2_1=tf.layers.conv2d(pooling1,32,(3,3),padding="same",activation=tf.nn.relu,name="conv2_1")
conv2_2=tf.layers.conv2d(conv2_1,32,(3,3),padding="same",activation=tf.nn.relu,name="conv2_2")

pooling2=tf.layers.max_pooling2d(conv2_2,(2,2),(2,2),name="pool2")

conv3_1=tf.layers.conv2d(pooling2,32,(3,3),padding="same",activation=tf.nn.relu,name="conv3_1")
conv3_2=tf.layers.conv2d(conv3_1,32,(3,3),padding="same",activation=tf.nn.relu,name="conv3_2")
pooling3=tf.layers.max_pooling2d(conv3_2,(2,2),(2,2),name="pool3")

flatten=tf.layers.flatten(pooling3)
y_=tf.layers.dense(flatten,10)

#way 1
# p_y=tf.nn.softmax(y_)
# y_one_hot=tf.one_hot(y,10,dtype=tf.float32)
# loss=tf.reduce_mean(tf.square(y_one_hot-p_y))


#way2

#y_->softmax
#y->one_hot
#loss=ylogy_
loss=tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)



predict=tf.argmax(y_,1)
correct_prediction=tf.equal(predict,y)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float64))






with tf.name_scope("train_op"):
    train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)

init=tf.global_variables_initializer()
batch_size=20
train_steps=100000

test_steps=100
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data,batch_labels=train_data.next_batch(batch_size)

        loss_val,accu_val,_=sess.run([loss,accuracy,train_op],feed_dict={x:batch_data,y:batch_labels})
        if (i+1)%500==0:
            print("step",i,loss_val,accu_val)

        if (i+1)%5000==0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val=[]
            for j in range(test_steps):
                test_batch_data,test_batch_labels=test_data.next_batch(batch_size)
                test_acc_val=sess.run([accuracy],feed_dict={x:test_batch_data,y:test_batch_labels})
                all_test_acc_val.append(test_acc_val)
            test_acc=np.mean(all_test_acc_val)
            print("test",i+1,test_acc)





