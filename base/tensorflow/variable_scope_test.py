import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

def test1():
    tf.compat.v1.reset_default_graph()
    ###tf.Variable()
    v1 = tf.Variable([2,2],dtype=tf.float32)
    print(v1.name)
    v1 = tf.Variable([2,2],dtype=tf.float32)
    print(v1.name)
    v1 = tf.Variable([2,2],dtype=tf.float32, name='V')
    print(v1.name)
    v1 = tf.Variable([2,2],dtype=tf.float32, name='V')
    print(v1.name)

    ###tf.get_variable()
    v2 = tf.compat.v1.get_variable(name='GetV',shape=[2,2])
    print(v2.name)
    # 必须指定变量名称，如果重名会报错
    v2 = tf.compat.v1.get_variable(name='GetV',shape=[2,2])
    print(v2.name)


def test2():
    print(np.arange(10))


if __name__ == "__main__":
    test2()

