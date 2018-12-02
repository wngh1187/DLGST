import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
import pandas as pd
from pandas import DataFrame

np.random.seed(20160512)

n0, mu0, variance0 = 800, [10,11],20         #개수, [x1,x2] 각각의 평균, 분산
data0 = multivariate_normal(mu0,np.eye(2)*variance0,n0)     #다변수 정규분포함수
df0 = DataFrame(data0,columns=['x','y'])          #data를 엑셀형식으로 변한다.
df0['t'] = 0
            # t=0인 데이터 를 난수로 만듬

n1, mu1, variance1 = 600,[18,20],22
data1 = multivariate_normal(mu1,np.eye(2)*variance1,n1)
df1 = DataFrame(data1,columns=['x','y'])
df1['t'] = 1
            # t=1인 데이터 를 난수로 만듬

df = pd.concat([df0,df1], ignore_index=True)    #t가 0,1인 데이터를 하나로 모은다.
df= df.reindex(permutation(df.index)).reset_index(drop=True)        #순서를 무작위로 만든다.

num_data = int(len(df)*0.8)
train_set = df[:num_data]           #전체 데이터의 80%만 학습 시키고
test_set = df[num_data:]            #20%는 테스트 한다.

train_x = train_set[['x','y']].as_matrix()        #학습데이터의 데이터값을 따로 저장한다.
train_t = train_set['t'].as_matrix().reshape([len(train_set),1])    #학습데이터의 타겟값을 따로 저장한다.
test_x = test_set[['x','y']].as_matrix()
test_t = test_set['t'].as_matrix().reshape([len(test_set),1])

x = tf.placeholder(tf.float32,[None,2])     #학습데이터를 저장하기위해 변수 x를 선언
w = tf.Variable(tf.zeros([2,1]))            #weight값을 저장하기 위해 변수 w를 선언
w0 = tf.Variable(tf.zeros([1]))             #basis를 저장하기위해 변수 w0를 선언
f = tf.matmul(x,w) + w0
p = tf.sigmoid(f)               #activation함수로 sigmoid 함수를 사용

t = tf.placeholder(tf.float32, [None,1])
loss = -tf.reduce_sum(t*tf.log(p) + (1-t)*tf.log(1-p))
train_step = tf.train.AdamOptimizer().minimize(loss)            #

correct_prediction = tf.equal(tf.sign(p-0.5),tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

train_accuracy = []
test_accuracy = []

#for i in range(1, 20001):
for _ in range(2500):
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    acc_val =sess.run(accuracy, feed_dict={x:train_x, t:train_t})
    train_accuracy.append(acc_val)
    acc_val = sess.run(accuracy,feed_dict={x:test_x, t:test_t})
    test_accuracy.append(acc_val)

w0_val, w_val = sess.run([w0,w])
w0_val, w1_val, w2_val = w0_val[0],w_val[0][0], w_val[1][0]
print(w0_val, w1_val, w2_val)

train_set0 = train_set[train_set['t'] == 0]
train_set1 = train_set[train_set['t'] == 1]

fig = plt.figure(figsize=(6,6))
subplot = fig.add_subplot(1,1,1)

subplot.plot(range(len(train_accuracy)), train_accuracy, linewidth=2, label='Trainin set')
subplot.plot(range(len(test_accuracy)), test_accuracy, linewidth=2, label='Test set')
subplot.legend(loc='upper left')
plt.show()

