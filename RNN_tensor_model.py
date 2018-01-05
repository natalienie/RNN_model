import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

'''
since i am trying to use 128 sample of price + macd to figure out the label,
in my opinion the chunk_size is 128 * 2 here.
however i still don't know if i need to flat this (128, 2) out to (256)
i would not do it this time ....
just treat it as a 128 chunks with chunk_size of 2


'''
hm_epochs = 3
n_classes = 3
batch_size = 128
chunk_size = 2
n_chunks = 128
rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

'''
no need to reshape in next_batch func anymore( no flatten out)
'''

def next_batch(data, i, batch_size):

    return np.array(data[i*batch_size:(i+1)*batch_size])


def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'bias': tf.Variable(tf.random_normal([n_classes]))}
   '''
   i am not sure whether the below 3 lines are necessary,
   to reshape the array into form acceptable to rnn_cells
   since the data i am going to feed is in already in this shape
   '''
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            hm_batch = int(len(x)/batch_size)
            for i in range(hm_batch):
                epoch_x = next_batch(X_train, i, batch_size)
                epoch_y = next_batch(y_train, i, batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoth_loss += c
                i += 1
            print('Epoch', epoch, 'completed out of', hm_epochs,'loss:' , epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))
