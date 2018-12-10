import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os

np.random.seed(134235252)
tf.set_random_seed(34413352)

mnist = input_data.read_data_sets("\\tmp\\data\\", one_hot=True)

num_filter1 = 20            #첫번째 필터의 개수

x = tf.placeholder(tf.float32, [None,784])
x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,num_filter1], stddev=0.2))     #필터사이즈xinput개수xoutput개수
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME')     #사이즈가 28x28인 이미지 1개를 5x5 필터 20장을 곱하여 20개의 데이터를 출력한다.
b_conv1 = tf.Variable(tf.constant(0.2, shape=[num_filter1]))                    #0이 아닌 0.1로 초기화를 시켜 파라미터 최적화를 효율적으로 한다.
h_conv1_cutoff = tf.nn.sigmoid(h_conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding= 'SAME')       #conv한 결과를 pool레이어에 넣어 14x14 사이즈의 이미지로 변환시킨다.

num_filter2 = 40             #두번째 필터의 개수

W_conv2 = tf.Variable(tf.truncated_normal([5,5,num_filter1,num_filter2], stddev=0.2))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME')         #사이즈가 14x14인 이미지 20개를 5x5 필터 40장을 곱하여 40개의 데이터를 출력한다.
b_conv2 = tf.Variable(tf.constant(0.2, shape=[num_filter2]))
h_conv2_cutoff = tf.nn.sigmoid(h_conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding= 'SAME')       #conv한 결과를 pool레이어에 넣어 7x7 사이즈의 이미지로 변환시킨다.

num_filter3 = 80             #세번째 필터의 개수

W_conv3 = tf.Variable(tf.truncated_normal([5,5,num_filter2,num_filter3], stddev=0.2))
h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='SAME')         #사이즈가 7x7인 이미지 20개를 5x5 필터 40장을 곱하여 40개의 데이터를 출력한다.
b_conv3 = tf.Variable(tf.constant(0.2, shape=[num_filter3]))
h_conv3_cutoff = tf.nn.sigmoid(h_conv3 + b_conv3)
h_pool3 = tf.nn.max_pool(h_conv3_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding= 'SAME')       #conv한 결과를 pool레이어에 넣어 4x4 사이즈의 이미지로 변환시킨다..

h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*num_filter3])       #[1 x (4*4*80)]인 형태로 변환시킨다.

num_unit1 = 4*4*num_filter3     #
num_unit2 = 1400                #첫번째 히든레이어의 노드 개수
num_unit3 = 700                 #두번째 히든레이어의 노드 개수


#첫번째 히든레이어
w2 = tf.Variable(tf.truncated_normal([num_unit1, num_unit2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_unit2]))
hidden2 = tf.nn.sigmoid(tf.matmul(h_pool2_flat,w2) + b2)       #Activation function으로 sigmoid 함수를 사용한다.

keep_prob2 = tf.placeholder(tf.float32)              #잘라내지 않고 남겨 둘 노드의 비율울 0~1 사이의 실수값으로 지정
hidden2_drop = tf.nn.dropout(hidden2, keep_prob2)   #dropout 계층을 추가하여 오버피팅을 방지한다.


#두번째 히든레이어
w1 = tf.Variable(tf.zeros([num_unit2,num_unit3]))
b1 = tf.Variable(tf.zeros([num_unit3]))
hidden1 = tf.nn.sigmoid(tf.matmul(hidden2_drop,w1) + b1)        #Activation function으로 sigmoid 함수를 사용한다.

keep_prob1 = tf.placeholder(tf.float32)             #잘라내지 않고 남겨 둘 노드의 비율울 0~1 사이의 실수값으로 지정
hidden1_drop = tf.nn.dropout(hidden1, keep_prob1)   #dropout 계층을 추가하여 오버피팅을 방지한다.


#Output Layer
w0 = tf.Variable(tf.zeros([num_unit3,10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden1_drop,w0) + b0)      #0~9 로 분리하기위해 sigmoid function을 이용하여 10개의 확률로 출력한다.

t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t*tf.log(p))                #categorical CE
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)      #Optimizer로 Adam을 사용한다.
correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


#Saver객체를 이용하여 파라미터값을 저장한다.
SAVER_DIR = "model"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)


if ckpt and ckpt.model_checkpoint_path:
  saver.restore(sess, ckpt.model_checkpoint_path)
'''

for i in range(100001):
    batch_xs, batch_ts = mnist.train.next_batch(50)       #batch size를 50으로 지정
    sess.run(train_step, feed_dict={x:batch_xs, t:batch_ts, keep_prob2 :0.6, keep_prob1 :0.5})    #첫번째 dropout층의 비율을 0.6, 두번째 dropout층의 비율을 0.5로 지정

    if i % 1000 == 0:
        loss_vals, acc_vals = [],[]
        for c in range(4):
            start = int(len(mnist.test.labels)/ 4*c)
            end = int(len(mnist.test.labels)/ 4*(c+1))
            # 테스트 세트 데이터를 분할해서 4회로 나누는 이유는 메모리 사용률을 줄이기 위함이다
            loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: mnist.test.images[start:end], t: mnist.test.labels[start:end], keep_prob2: 1.0,keep_prob1: 1.0})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
        loss_val = np.sum(loss_vals)
        acc_val = np.mean(acc_vals)
        print('Step: %d, Loss: %f, Accuracy: %f' %(i, loss_val, acc_val))
        saver.save(sess, checkpoint_path, global_step=i)
'''

#첫 번째 단계의 필터를 적용한 이미지를 출력한다
batch_xs, batch_ts = mnist.train.next_batch(50)
conv1_vals, cutoff1_vals = sess.run( [h_conv1, h_conv1_cutoff], feed_dict={x: batch_xs,keep_prob2:1.0 ,keep_prob1:1.0})
#Placeholder x에 변수 image의 내용을 저장한 상태이다.
#h_conv1과 h_conv1_cutoff는 첫 번째 단계의 필터를 적용한 결과를 나타낸다.
#h_conv1_cutoff는 ReLU함수를 이용해 일정 값보다 작은 픽셀값을 잘라 내었다.

fig = plt.figure(figsize=(16,4))

for f in range(num_filter1):
  subplot = fig.add_subplot(4, 10, f+1)
  subplot.set_xticks([])
  subplot.set_yticks([])
  subplot.imshow(conv1_vals[0, :, :, f], cmap = plt.cm.gray_r, interpolation = 'nearest')
for f in range(num_filter1):
  subplot = fig.add_subplot(4, 10, num_filter1 + f + 1)
  subplot.set_xticks([])
  subplot.set_yticks([])
  subplot.imshow(cutoff1_vals[0, :, :, f], cmap = plt.cm.gray_r, interpolation = 'nearest')
  #32가지 필터에 해당하는 32가지 이미지가 출력되고 있다.

  # 두 번째 단계의 필터를 적용한 이미지를 출력한다.
  conv2_vals, cutoff2_vals = sess.run([h_conv2, h_conv2_cutoff], feed_dict={x: batch_xs,keep_prob2: 1.0, keep_prob1: 1.0})
  # h_conv2과 h_conv2_cutoff는 두 번째 단계의 필터를 적용한 결과를 나타낸다.
  # h_conv2_cutoff는 ReLU함수를 이용해 일정 값보다 작은 픽셀값을 잘라 내었다.

fig = plt.figure(figsize=(16, 8))

for f in range(num_filter2):
  subplot = fig.add_subplot(8, 10, f + 1)
  subplot.set_xticks([])
  subplot.set_yticks([])
  subplot.imshow(conv2_vals[0, :, :, f], cmap=plt.cm.gray_r, interpolation='nearest')
for f in range(num_filter2):
  subplot = fig.add_subplot(8, 10, num_filter2 + f + 1)
  subplot.set_xticks([])
  subplot.set_yticks([])
  subplot.imshow(cutoff2_vals[0, :, :, f], cmap=plt.cm.gray_r, interpolation='nearest')


#세 번째 단계의 필터를 적용한 이미지를 출력한다.
conv3_vals, cutoff3_vals = sess.run( [h_conv3, h_conv3_cutoff], feed_dict={x: batch_xs,keep_prob2:1.0, keep_prob1:1.0})
#h_conv2과 h_conv2_cutoff는 두 번째 단계의 필터를 적용한 결과를 나타낸다.
#h_conv2_cutoff는 ReLU함수를 이용해 일정 값보다 작은 픽셀값을 잘라 내었다.

fig = plt.figure(figsize=(16,8))

for f in range(num_filter3):
  subplot = fig.add_subplot(8, 20, f+1)
  subplot.set_xticks([])
  subplot.set_yticks([])
  subplot.imshow(conv3_vals[0,:,:,f], cmap=plt.cm.gray_r, interpolation='nearest')
for f in range(num_filter3):
  subplot = fig.add_subplot(8, 20, num_filter3+f+1)
  subplot.set_xticks([])
  subplot.set_yticks([])
  subplot.imshow(cutoff3_vals[0,:,:,f], cmap=plt.cm.gray_r, interpolation='nearest')

#1.필터수 를 늘렸다. 마지막에는 4x4
#2. 필터수보다 레이어의 수가 많으면 큰 의미가 없는것 같다.
plt.show()