import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle

np.random.seed(12314124)
tf.set_random_seed(124123)

with open('ORENIST.data', 'rb') as file:
    images, labels = pickle.load(file, encoding="bytes")        # 책 대로 하면 오류가 뜬다.

def edge_filter():
    filter0 = np.array(     #세로 강조
        [[2,1,0,-1,-2],
         [3,2,0,-2,-3],
         [4,3,0,-3,-4],
         [3,2,0,-2,-3],
         [2,1,0,-1,-2]])/23.0
    filter1 = np.array(     #가로 강조
        [[2,3,4,3,2],
         [1,2,3,2,1],
         [0,0,0,0,0],
         [-1,-2,-3,-2,-1],
         [-2,-3,-4,-3,-2]])/23.0

    filter_array = np.zeros([5,5,1,2])
    filter_array[:,:,0,0] = filter0
    filter_array[:,:,0,1] = filter1         #filter들을 1x2부분에 저장

    return tf.constant(filter_array, dtype = tf.float32)

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

W_conv = tf.Variable(tf.truncated_normal([5,5,1,2], stddev = 0.1))          # Variable로 지정하여 필터를 최적화 한다.
h_conv = tf.abs(tf.nn.conv2d(x_image, W_conv, strides=[1,1,1,1], padding = 'SAME'))
h_conv_cutoff = tf.nn.relu(h_conv-0.2)

h_pool = tf.nn.max_pool(h_conv_cutoff, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# pooling layer의 결과로 14x14 데이터가 2개가 생김

h_pool_flat = tf.reshape(h_pool, [-1, 392])     # 14x14x2 = 392
num_units1 = 392
num_units2 = 3

w2 = tf.Variable(tf.truncated_normal([num_units1,num_units2]))      #392x2
b2 = tf.Variable(tf.zeros(num_units2))
hidden2 = tf.nn.relu(tf.matmul(h_pool_flat,w2) + b2)

w0 = tf.Variable(tf.zeros([num_units2,3]))
b0 = tf.Variable(tf.zeros([3]))
p = tf.nn.softmax(tf.matmul(hidden2,w0) + b0)

t = tf.placeholder(tf.float32, [None,3])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(501):
    sess.run(train_step, feed_dict={x:images, t:labels})
    if i % 50 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:images, t:labels})
        print('Step: %d, Loss: %f, Accuracy: %f' %(i, loss_val, acc_val))

hidden2_vals = sess.run(hidden2, feed_dict={x:images})          # 은닉계측
z1_vals = [[],[],[]]
z2_vals = [[],[],[]]
#0,1,2 이미지별 좌표를 입력한다.

for hidden2_val, label in zip(hidden2_vals, labels):
    label_num = np.argmax(label)
    z1_vals[label_num].append(hidden2_val[0])
    z2_vals[label_num].append(hidden2_val[1])

fig = plt.figure(figsize=(5,5))
subplot = fig.add_subplot(1,1,1)
subplot.scatter(z1_vals[0], z2_vals[0], s=200, marker="|")
subplot.scatter(z1_vals[1], z2_vals[1], s=200, marker="_")
subplot.scatter(z1_vals[2], z2_vals[2], s=200, marker="+")

filter_vals, conv_vals = sess.run([W_conv,h_conv_cutoff], feed_dict={x:images[:9]})         #변수 images에 준비해 둔 이미지 데이터 중 첫 9개를 placeholder에 저장

fig = plt.figure(figsize=(10,3))

for i in range(2):      #filter 출력
    subplot = fig.add_subplot(3,10,10*(i+1)+1)      #11, 21번째에 필터 출력
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(filter_vals[:,:,0,i], cmap=plt.cm.gray_r, interpolation='nearest')


for i in range(9):
    subplot = fig.add_subplot(3,10,i+2)         # 원본이미지 출력
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(images[i].reshape((28,28)), vmin= 0, vmax= 1, cmap=plt.cm.gray_r, interpolation='nearest')

    subplot = fig.add_subplot(3, 10, 10+i+2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(conv_vals[i, :, :,0], vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation='nearest')       #i번째 이미지와 첫번째 필터 곱의 데이터 출력

    subplot = fig.add_subplot(3,10, 20+i+2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(conv_vals[i,:,:,1], vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation='nearest')         #i번째 이미지와 두번째 필터 곱의 데이터 출력

pool_vals = sess.run(h_pool, feed_dict={x:images[:9]})  #이미지의 개수 x 이미지 크기(세로x가로) x 출력 레이어 개수

fig = plt.figure(figsize=(10,3))

for i in range(2):      #filter 출력
    subplot = fig.add_subplot(3,10,10*(i+1)+1)      #11, 21번째에 필터 출력
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(filter_vals[:,:,0,i], cmap=plt.cm.gray_r, interpolation='nearest')


for i in range(9):
    subplot = fig.add_subplot(3,10,i+2)         # 원본이미지 출력
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(images[i].reshape((28,28)), vmin= 0, vmax= 1, cmap=plt.cm.gray_r, interpolation='nearest')

    subplot = fig.add_subplot(3, 10, 10+i+2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(pool_vals[i, :, :,0], vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation='nearest')       #i번째 이미지와 첫번째 필터 곱의 데이터 출력

    subplot = fig.add_subplot(3,10, 20+i+2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(pool_vals[i,:,:,1], vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation='nearest')         #i번째 이미지와 두번째 필터 곱의 데이터 출력

plt.show()