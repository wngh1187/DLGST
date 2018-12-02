import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle

with open('ORENIST.data','rb') as file:
    images, labels = pickle.load(file, encoding="bytes")        # 책 대로 하면 오류가 뜬다.

'''
fig = plt.figure(figsize=(10,5))

for i in range(40):
    subplot = fig.add_subplot(4,10,i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(images[i].reshape(28,28), vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation='nearest')
'''

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

x = tf.placeholder(tf.float32, [None,784])      #28x28 = 784개의 픽셀로 된 이미지 데이터를 저장할 placeholder를 준비
x_image = tf.reshape(x,[-1,28,28,1])            # 첫번째 -1은 placeholder에 저장되어 있는 데이터 개수에 따라 적절한 크기로 조정

W_conv = edge_filter()          # 입력했던 필터를 가져온다.
h_conv = tf.abs(tf.nn.conv2d(x_image, W_conv, strides=[1,1,1,1], padding='SAME'))       # x_image에 필터를 적용한다.
h_conv_cutoff = tf.nn.relu(h_conv-0.2)      #필터의 효과를 강조한다. 0.2 보다 작은 값은 0으로 만든다.    #이미지수 x size(row,col) x output layer수

h_pool = tf.nn.max_pool(h_conv_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')       #픽셀 내 최대값을 출력한다.
#필터에서 출력된 28x28 픽셀 이미지를 2x2 픽셀 블록으로 분해해서 각각의 블록을 하나의 픽셀로 치환한다.
#28x28 -> 14x14 크기의 이미지로 변환된다.
#ksize 옵션으로 지정된 크기의 블록을 strides옵션으로 지정된 간격으로 이동시켜 가며, 블록 내에 있는 픽셀의 최대값으로 치환해간다.

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

filter_vals, conv_vals = sess.run([W_conv,h_conv_cutoff], feed_dict={x:images[:9]})         #변수 images에 준비해 둔 이미지 데이터 중 첫 9개를 placeholder에 저장

fig = plt.figure(figsize=(10,3))

for i in range(2):      #filter 출력
    subplot = fig.add_subplot(3,10,10*(i+1)+1)      #11, 21번째에 필터 출력
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(filter_vals[:,:,0,i], cmap=plt.cm.gray_r, interpolation='nearest')

v_max = np.max(conv_vals)

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

v_max = np.max(pool_vals)

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