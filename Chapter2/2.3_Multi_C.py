import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20181123)

mnist = input_data.read_data_sets("tmp/data/", one_hot=True)    #텐서플로에는 웹에서 공개하고 있는 MNIST 데이터 세트를 다운로드해서 NumPy의 array 오브젝트로 저장하는 모듈이 미리 준비되어있다.

x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784,10]))
w0 = tf.Variable(tf.zeros([10]))
f = tf.matmul(x,w) + w0
p = tf.nn.softmax(f)

t = tf.placeholder(tf.float32, [None,10])
loss = -tf.reduce_sum(t*tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(p,1),tf.argmax(t,1))        #tf.argmax는 복수의 요소가 나열된 리스트에서 최댓값을 갖는 요소의 인데스를 추출하는함수
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

images,labels = mnist.train.next_batch(10)      # 10개의 데이터를 추출

print(images[0])
print(labels[0])

fig = plt.figure(figsize=(8,4))
for c, (image, label) in enumerate(zip(images, labels)):        #zip함수를 이용하여 images와 labels을 병렬적으로 묶은 다음, enumerate를 이용하여 순서와 묶음을 c 와(image,label)에 넣는다.
    subplot = fig.add_subplot(2,5,c+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(label))  #정답레이블
    subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation= 'nearest')        #
