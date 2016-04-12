import tensorflow as tf


# with tf.variable_scope("foo") as foo_scope:
#     v = tf.get_variable("v", [1])
# with tf.variable_scope(foo_scope):
#     w = tf.get_variable("w", [1])
# with tf.variable_scope(foo_scope, reuse=True):
#     v1 = tf.get_variable("v", [1])
#     w1 = tf.get_variable("w", [1])
# print v1 == v
# print w1 == w


with tf.variable_scope("foo") as foo_scope:
    assert foo_scope.name == "foo"
with tf.variable_scope("bar"):
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo"  # Not changed.
