import os

print("env before", os.environ.get('TF_CPP_MIN_VLOG_LEVEL', ''))
os.environ['TF_CPP_MIN_VLOG_LEVEL']='1'
print("env after", os.environ.get('TF_CPP_MIN_VLOG_LEVEL', ''))

import tensorflow as tf

print("TensorFlow version: ", tf.__git_version__)

from tensorflow.contrib.compiler import jit
tf.reset_default_graph()
jit_scope = jit.experimental_jit_scope
with jit_scope(compile_ops=True):
    N = 500*1000*1000
    x = tf.Variable(tf.random_uniform(shape=(N,)))
    y = 0.1*x*x*x*x*x-0.5*x*x*x*x+.25*x*x*x+.75*x*x-1.5*x-2
    y0 = y[0]

import time
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(y.op)
start_time = time.time()
print(sess.run(y0))
end_time = time.time()
print("%.2f sec"%(end_time-start_time))
