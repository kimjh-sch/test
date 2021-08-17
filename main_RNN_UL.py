import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

tf.reset_default_graph()


N = 10
num_train = 100000
num_test = 10000
epochs = 300
batch_size = 1000
learning_rate = 0.0001

tx_snr = 0
noise_power = 10**(tx_snr/10.0)
noise_power = 1


load = sio.loadmat('data/Train_data_%d_%d.mat' % (N, num_train))
loadTest = sio.loadmat('data/Test_data_%d_%d.mat' % (N, num_test))
H_train_valid = load['Xtrain']
P_train_valid = load['Ytrain']
H_test = loadTest['Xtest']
P_test = loadTest['Ytest']
timeW = loadTest['swmmsetime']
swmmsetime = timeW[0, 0]


def network(input):
    num_units = 100
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=num_units) 
    #cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units) 
    #cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units) 
    #cell = tf.nn.rnn_cell.GRUCell(num_units=num_units) 
    outputs, states = tf.nn.dynamic_rnn(cell, input, dtype = tf.float32)
    out = tf.layers.dense(states, N)  # for RNN, GRU
    #out = tf.layers.dense(states[1], N)  # only for LSTM (states[0]: cell state, states[1]: hidden state)
    pred = tf.nn.sigmoid(out)

    return pred


x = tf.placeholder(tf.float32, [None, N, N])  # channel
y = tf.placeholder(tf.float32, [None, N])  # label power

P_pred = network(x)


def calc_sum_rate(user_num, power, abs_H, noise_power):
    abs_H = tf.reshape(abs_H, [-1, user_num, user_num])
    abs_H_2 = tf.square(abs_H)
    rx_power = tf.multiply(abs_H_2, tf.reshape(power, [-1, user_num, 1]))
    mask = tf.eye(user_num)
    valid_rx_power = tf.reduce_sum(tf.multiply(rx_power, mask), axis=1)
    interference = tf.reduce_sum(tf.multiply(rx_power, 1-mask), axis=1) + noise_power
    rate = tf.log(1 + tf.divide(valid_rx_power, interference)) / tf.log(2.0)
    sum_rate = tf.reduce_mean(tf.reduce_sum(rate, axis=1))
    return sum_rate

sum_rate_pred = calc_sum_rate(N, P_pred, x, noise_power)
cost = tf.negative(sum_rate_pred)
optimizer = tf.train.AdamOptimizer(learning_rate)
objective = optimizer.minimize(cost)



valid_split = 0.1

total_sample_size = num_train
valid_sample_size = int(total_sample_size*valid_split)
train_sample_size = total_sample_size - valid_sample_size

H_train = H_train_valid[ 0:train_sample_size, :, :]
P_train = P_train_valid[0:train_sample_size, :]

H_valid = H_train_valid[ train_sample_size:total_sample_size, :, :]
P_valid = P_train_valid[train_sample_size:total_sample_size, :]


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
start_time = time.time()

cost_history = np.zeros((epochs, 3))
total_batch = int(total_sample_size / batch_size)

start_time = time.time()
for epoch in range(epochs):
    for ii in range(total_batch):
        batch = np.random.randint(train_sample_size, size=batch_size)
        _, train_cost = sess.run([objective, cost], feed_dict={x: H_train[batch, :, :], y: P_train[batch, :] })

    valid_cost = sess.run(cost, feed_dict={x: H_valid, y: P_valid})
    
    cost_history[epoch, 0] = epoch + 1
    cost_history[epoch, 1] = train_cost
    cost_history[epoch, 2] = valid_cost

    if (epoch % 5 == 0):
        print('\n Epochs = %d, ' % epoch, ' Train cost = %f, ' % (train_cost), ' Valid cost = %f, ' % (valid_cost), ' Time = %f ' % (time.time() - start_time), )
    else:
        print("#", end="")
train_time = time.time()-start_time


#### for Test ####
H_t = np.reshape(H_test, (num_test, N, N))
P_t = np.reshape(P_test, (num_test, N))

start_time = time.time()
pred_power = sess.run(P_pred, feed_dict={x: H_t, y: P_t})
pred_time = time.time()-start_time
pred_power = np.reshape(pred_power, (num_test, N))


def IC_sum_rate(H, p, noise_var):
    H = np.square(H)
    fr = np.diag(H)*p
    ag = np.dot(H,p) + noise_var - fr
    y = np.sum(np.log(1+fr/ag) )
    return y

def np_sum_rate(X, Y, noise_var):
    avg = 0
    n = X.shape[0]
    for i in range(n):
        avg += IC_sum_rate(X[i,:,:],Y[i,:], noise_var)/n
    return avg

sum_rate_rnn = np_sum_rate(H_test, pred_power, noise_power)*np.log2(np.exp(1))
sum_rate_swmmse = np_sum_rate(H_test, P_test, noise_power)*np.log2(np.exp(1))

print()
print('Training time = %f' % train_time)
print('sum rate for RNN(UL) = %f, Test time = %f' % (sum_rate_rnn, pred_time))
print('sum rate for SWMMSE = %f, Test time = %f' % (sum_rate_swmmse, swmmsetime))
print("RNN/SWMMSE = %f%%" % (sum_rate_rnn/sum_rate_swmmse*100))


pf = open('RNN_UL_N%d_e%d_b%d_lr%s.txt' %(N, epochs, batch_size, f"{learning_rate:1.0E}"), 'w')
pf.write('N = %d, epochs = %d, batch_size = %d, learning_rate = %f\n' % (N, epochs, batch_size, learning_rate))
pf.write('Training time = %f\n' % train_time)
pf.write('sum rate for RNN(UL) = %f, Test time = %f\n' % (sum_rate_rnn, pred_time))
pf.write('sum rate for SWMMSE = %f, Test time = %f\n' % (sum_rate_swmmse, swmmsetime))
pf.write("RNN/SWMMSE = %f%%" % (sum_rate_rnn/sum_rate_swmmse*100))
pf.close()


#### for figure ####
sio.savemat('RNN_UL_N%d_e%d_b%d_lr%s.mat' %(N, epochs, batch_size, f"{learning_rate:1.0E}"), {'cost_history': cost_history})
ax = plt.figure().gca()
plt.plot(range(epochs), cost_history[:, 1], label='Train cost', color='darkred', linewidth=1, marker='x', markersize=5, markerfacecolor='none')
plt.plot(range(epochs), cost_history[:, 2], label='Valid cost', color='darkblue', linewidth=1, marker='o', markersize=5, markerfacecolor='none')
plt.legend(loc='upper right')
plt.xlabel('epochs')
plt.ylabel('Cost')
plt.title('RNN UL')
plt.grid(linestyle='--', color='lavender')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.savefig('RNN_UL_N%d_e%d_b%d_lr%s.png' %(N, epochs, batch_size, f"{learning_rate:1.0E}"))
plt.show()
