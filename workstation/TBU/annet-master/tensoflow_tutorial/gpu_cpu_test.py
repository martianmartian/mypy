#encoding:UTF-8
import tensorflow as tf


# GPU
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # 新建session with log_device_placement并设置为True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # 运行这个 op.
# print sess.run(c)


# # 新建一个graph.
# with tf.device('/cpu:0'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # 新建session with log_device_placement并设置为True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # 运行这个op.
# print sess.run(c)


# 多GPU
# 新建一个 graph.
with tf.device('/gpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# 新建 session with log_device_placement 并设置为 True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 运行这个 op.
print sess.run(c)