import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
import pandas as pd
from pandas import DataFrame

np.random.seed(20149200)
tf.set_random_seed(167100203)

def generate_datablock(n,mu,var,t):
    data = multivariate_normal(mu,np.eye(2)*var, n)
    df = DataFrame(data,columns=['x1','x2'])
    df['t'] = t
    return df

df0 = generate_datablock(15,[7,7],22,0)
df1 = generate_datablock(15,[22,7],22,0)
df2 = generate_datablock(10,[7,22],22,0)
df3 = generate_datablock(25,[22,22],22,1)

df= pd.concat([df0,df1,df2,df3], ignore_index= True)
train_set = df.reindex(permutation(df.index)).reset_index(drop=True)

train_x = train_set[['x1','x2']].as_matrix()
train_t = train_set['t'].as_matrix().reshape([len(train_set),1])

num_units = 4  # 은닉계층의 노드수
mult = train_x.flatten().mean()     #트레이닝 데이터의 평균값

x =  tf.placeholder(tf.float32,[None,2])

w1 = tf.Variable(tf.truncated_normal([2,num_units]))    #지정된 크기의 다차원 리스트에 해당하는 Variable을 준비해서 각각의 요소를 평균 0, 표준편차 1인 정규분포를 따르는 난수로 초기화
                                                        #은닉계층의 경우 초기값을 0이 아닌 난수로 초기화 시킨다. 0으로 초기화하면 경사하강법에 의한 최적화가 진행되지 않는 경우도 있음
b1 = tf.Variable(tf.zeros([num_units]))
hidden1 = tf.nn.tanh(tf.matmul(x,w1) + b1*mult)         #mult를 곱해줌으로써 파라미터 최적화 속도를 고속화 시킨다.

w0 = tf.Variable(tf.zeros([num_units,1]))
b0 = tf.Variable(tf.zeros([1]))
p = tf.nn.sigmoid(tf.matmul(hidden1,w0) + b0*mult)

t = tf.placeholder(tf.float32,[None,1])
loss = -tf.reduce_sum(t*tf.log(p) + (1-t)*tf.log(1-p))          #output layer의 activation function이 sigmoid이기 때문에 binary cee를 사용
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(4001):
    sess.run(train_step, feed_dict={x:train_x, t: train_t})
    if i%100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy] ,feed_dict={x:train_x, t:train_t})
        print('Step: %d, Loss: %f, Accuracy: %f' %(i,loss_val,acc_val))

train_set1 = train_set[train_set['t'] == 1]
train_set2 = train_set[train_set['t'] == 0]

fig = plt.figure(figsize=(6,6))
subplot = fig.add_subplot(1,1,1)
subplot.set_ylim([0,30])
subplot.set_xlim([0,30])
subplot.scatter(train_set1.x1, train_set1.x2, marker='x')
subplot.scatter(train_set2.x1, train_set2.x2, marker='o')

locations = []                                              #100x100 영역으로 분할
for x2 in np.linspace(0,30,100):
    for x1 in np.linspace(0,30,100):
        locations.append((x1,x2))

p_vals = sess.run(p,feed_dict={x:locations})
p_vals = p_vals.reshape((100,100))
subplot.imshow(p_vals, origin='lower', extent=(0,30,0,30), cmap= plt.cm.gray_r, alpha= 0.5)
plt.show()