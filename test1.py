# -*- coding:utf-8 -*-


import tensorflow as tf


# normal = tf.truncated_normal([4, 4, 4], mean=0.0, stddev=1.0)
#
# a = tf.Variable(tf.random_normal([2,2],seed=1))
# b = tf.Variable(tf.truncated_normal([2,2],seed=2))
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(a))
#     print(sess.run(b))

n1 = tf.constant(["1234","6789"])
n2 = tf.string_to_number(n1,out_type=tf.float32)

sess = tf.Session()

result = sess.run(n2)
print (result)

sess.close()
