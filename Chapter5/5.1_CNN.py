import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(134235252)
tf.set_random_seed(34413352)

mnist = input_data.read_data_sets("\\tmp\\data\\", one_hot=True)

num_filter1 = 32

x = tf.placeholder(tf.float32, [None,784])
x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,num_filter1], stddev=0.1))
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME')     #28x28이미지에 5x5 필터 32장을 곱하여 32개의 데이터를 출력한다.
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filter1]))                    #0이 아닌 0.1로 초기화를 시켜 파라미터 최적화를 효율적으로 한다.
h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding= 'SAME')       #conv한 결과를 pool레이어에 넣어 14x14 사이즈의 이미지로 변환시킨다.

num_filter2 = 64

W_conv2 = tf.Variable(tf.truncated_normal([5,5,num_filter1,num_filter2], stddev=0.1))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME')         #32개의 이미지를 64개의 필터에 넣어 64개의 데이터를 출력한다.
b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filter2]))
h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding= 'SAME')       #conv한 결과를 pool레이어에 넣어 7x7 사이즈의 이미지로 변환시킨다.

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filter2])       #1 x 7*7*64인 형태로 변환시킨다.

num_unit1 = 7*7*num_filter2
num_unit2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_unit1, num_unit2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_unit2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat,w2) + b2)       #1024개의 데이터

keep_prob = tf.placeholder(tf.float32)          #잘라내지 않고 남겨 둘 노드의 비율울 0~1 사이의 실수값으로 지정
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_unit2,10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden2_drop,w0) + b0)      #0~9 로 분리하기위해 10개의 데이터로 출력한다.

t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t*tf.log(p))          #categorical CE
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(20000):
    batch_xs, batch_ts = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x:batch_xs, t:batch_ts, keep_prob :0.5})

    if i % 500 == 0:
        loss_vals, acc_vals = [],[]
        for c in range(4):
            start = int(len(mnist.test.labels)/ 4*c)
            end = int(len(mnist.test.labels)/ 4*(c+1))
            loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: mnist.test.images[start:end], t: mnist.test.labels[start:end], keep_prob: 1.0})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
        loss_val = np.sum(loss_vals)
        acc_val = np.mean(acc_vals)
        print('Step: %d, Loss: %f, Accuracy: %f' %(i, loss_val, acc_val))
        saver.save(sess, '\\tmp\\cnn_session', global_step=1)


