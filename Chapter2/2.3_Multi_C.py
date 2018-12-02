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
loss = -tf.reduce_sum(t*tf.log(p))      # categorical CEE
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(p,1),tf.argmax(t,1))        # tf.argmax는 복수의 요소가 나열된 리스트에서 최댓값을 갖는 요소의 인데스를 추출하는함수, 1 은 가로 방향, 0은 세로 방향
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast 를 이용하여 Bool 값을 1,0으로 변환

sess = tf.InteractiveSession()      #대화형 파이썬에서 사용하는 세션함수
sess.run(tf.global_variables_initializer())
i = 0;

for _ in range(2000) :
    i += 1
    batch_xs ,batch_ts  = mnist.train.next_batch(100)       #트레이닝 세트에서 100개의 데이터를 추출, 데이터를 어디까지 추출 했는지 기억하고, 호출 할때 마다 다음 데이터를 추출하는 역할
    sess.run(train_step, feed_dict={x:batch_xs, t:batch_ts})
   # if i % 100 == 0:
    #    loss_val, acc_val = sess.run([loss, accuracy],feed_dict={x:mnist.test.images, t: mnist.test.labels})    #각각 테스트 세트가 가진 모든 이미지 데이터와 라벨을 포함하는 리스트로 되어있다.


images, labels = mnist.test.images, mnist.test.labels
p_val = sess.run(p, feed_dict={x:images, t:labels})

fig = plt.figure(figsize=(8,15))
for i in range(10):
    c = 1
    for (image, label, pred) in zip(images,labels,p_val):
        prediction, actual = np.argmax(pred), np.argmax(label)
        if prediction != i:     # 예상한 값이 0~9 까지 를 확인
            continue
        if (c < 4 and i == actual) or (c>= 4 and i != actual):      # 왼쪽 3개는 정답, 오른쪽 3개는 오류
            subplot = fig.add_subplot(10,6,i*6+c)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title('%d / %d' % (prediction, actual))
            subplot.imshow(image.reshape(28,28), vmin=0, vmax =1, cmap=plt.cm.gray_r, interpolation="nearest")
            c += 1
            if c> 6:
                break
plt.show()