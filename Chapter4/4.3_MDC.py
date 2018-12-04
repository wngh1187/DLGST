import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(2014415)
tf.set_random_seed(321531)

mnist = input_data.read_data_sets("\\tmp\data", one_hot = True)

num_filters = 16        #필터 개수

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

W_conv = tf.Variable(tf.truncated_normal([5,5,1,num_filters], stddev = 0.1))        #stddev 로 난수의 범위를 조절
h_conv = tf.nn.conv2d(x_image,W_conv,strides = [1,1,1,1], padding = 'SAME')         #OCE처럼 에지를 추출하는것이 아니라 '이미지 분류에 최적인 특징을 추출하는것이 목적, 그러므로 음수도 의미가 있다.
h_pool = tf.nn.max_pool(h_conv, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

h_pool_falt = tf.reshape(h_pool, [-1, 14*14*num_filters])   #필터의 개수x14x14개 만큼 데이터가 있다.

num_units1 = 14*14*num_filters
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1,num_units2]))
b2 =tf.Variable(tf.zeros(num_units2))
hidden2 = tf.nn.relu(tf.matmul(h_pool_falt,w2) + b2)

w0 = tf.Variable(tf.zeros([num_units2,10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden2,w0) + b0)      #0~9까지 10개의 데이터가 나온다

t = tf.placeholder(tf.float32, [None,10])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss) # 층이 복잡하기 때문에 learning late를 낮춘다.
correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

#saver.restore(sess, '\\tmp\\mdc_session-4000')

for i in range(4001):
    batch_xs, batch_ts = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, t:batch_ts})
    if i %400 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict = {x:mnist.test.images, t:mnist.test.labels})
        print('Step: %d, Loss :%f, Accuracy : %f' %(i,loss_val,acc_val))
        saver.save(sess, '\\tmp\\mdc_session', global_step = i)     #mdc_session-<처리횟수>, mdc_session-<처리횟수>.meta라는 파일 생성



