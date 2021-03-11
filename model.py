import tensorflow.compat.v1 as tf
import numpy as np
import data as dt

tf.disable_eager_execution()


def get_computational_graphs():
    cls_num = 10
    pixel_num = 28*28
    X = tf.placeholder(tf.float32, [None, pixel_num],name= 'X' )
    target = tf.placeholder(tf.float32, [None, cls_num], name = 'target')

    W = tf.Variable(tf.zeros([pixel_num, cls_num]))
    b = tf.Variable(tf.zeros([cls_num]))
    logits = tf.matmul(X,W) + b

    pred = tf.nn.softmax( logits )
    pred = tf.argmax(pred,axis= -1 )

    loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                                   labels = target)
    cost = tf.reduce_mean(loss)
    return pred, cost, X, target, W

def optimize(X,Y,cost, inp, target,  batch_size = 128, itr = 1000):
    opt = tf.train.AdamOptimizer().minimize(cost)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for _ in range(itr):
        indx = np.random.randint(0,X.shape[0],[batch_size])
        feed_dict = {inp:X[indx],
                     target: Y[indx]}
        sess.run(opt,feed_dict)
    return sess

def get_acc (model,inp,  X, Y_ac, sess) :
    Y_act1 = tf.placeholder(tf.float32,[None, 10])
    Y_act = tf.argmax(Y_act1,axis=1)

    acc = tf.equal(model,Y_act)
    acc= tf.reduce_mean(tf.cast(acc,tf.float32))
    acc = sess.run(acc,{inp:X, Y_act1: Y_ac})
    return acc


if __name__ == '__main__':
    train, test = dt.get_prepro_imgs()
    pred, cost, X, Y, W= get_computational_graphs()
    sess = optimize(train[0], train[1], cost, X,Y )
    acc = get_acc(pred,X, test[0],
                  test[1], sess)
    print(acc)
    W = sess.run(W)
    dt.plt_imgs(np.transpose(W),range(12),None,(2,5), intr = lambda X: X)