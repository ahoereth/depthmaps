
# coding: utf-8

# # Demo of the Make3D dataset
# http://make3d.cs.cornell.edu/data.html#make3d

# In[ ]:


import tensorflow as tf
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from Make3D import train_pairs, test_pairs


# # Visualize samples from the dataset

# In[ ]:


_, axis = plt.subplots(5, 2, figsize=(10,20))
plt.tight_layout()
for (rgb, d), (ax1, ax2) in zip(train_pairs[:10], axis):
    ax1.axis('off'), ax2.axis('off')
    ax1.imshow(rgb)
    ax2.imshow(sp.misc.imresize(d, rgb.shape))
plt.show()


# # Take a naive convolutional network approach

# ## Simplify dataset first
# - Convert to grayscale
# - Scale targets down so the convolutional network can use striding and so we do not need padding
# - Normalize values

# In[ ]:


train_data, train_targets = zip(*train_pairs)
test_data, test_targets = zip(*test_pairs)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

train_data = [sp.misc.imresize(rgb2gray(img), (128, 96))/127.5 - 1 for img in train_data]
train_targets = [sp.misc.imresize(img, (47, 31))/127.5 - 1 for img in train_targets]
test_data = [sp.misc.imresize(rgb2gray(img), (128, 96))/ 127.5 - 1 for img in test_data]
test_targets = [sp.misc.imresize(img, (47, 31))/127.5 - 1 for img in test_targets]

train_x, train_t = np.asarray(train_data), np.asarray(train_targets)
test_x, test_t = np.asarray(test_data), np.asarray(test_targets)

print('train input/target shapes', train_data[0].shape, train_targets[0].shape)
print('train input min/max/ptp', np.min(train_data), np.max(train_data), np.ptp(train_data))
print('train target min/max/ptp', np.min(train_targets), np.max(train_targets), np.ptp(train_targets))

tuples = zip(train_x[:10], train_t[:10])
fig, axis = plt.subplots(5, 2, figsize=(10,20))
plt.tight_layout(), plt.gray()
for (rgb, d), (ax1, ax2) in zip(tuples, axis):
    ax1.axis('off'), ax2.axis('off')
    ax1.imshow(rgb)
    ax2.imshow(sp.misc.imresize(d, rgb.shape))
plt.show()


# ## Define Tensorflow Graph

# In[ ]:


# MODEL
x = tf.placeholder(tf.float32, (None, 128, 96))
t = tf.placeholder(tf.float32, (None, 47, 31))

x_ = tf.reshape(x, (-1, 128, 96, 1))
net = tf.layers.conv2d(x_, filters=16, kernel_size=16, strides=2, activation=tf.nn.relu)
net = tf.layers.conv2d(net, filters=16, kernel_size=8, strides=1, activation=tf.nn.relu)
net = tf.layers.conv2d(net, filters=32, kernel_size=4, strides=1, activation=tf.nn.relu)
net = tf.layers.conv2d(net, filters=1, kernel_size=1)

y = tf.squeeze(net, axis=3)

loss = tf.reduce_mean(tf.square(t - y))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)


# ## Train
# Regularily evaluate the loss on the test data and compute test predictions when done with training.
# 
# **10000 epochs is nothing one wants to run on the CPU.**

# In[ ]:


def batches(x, y, batchsize=32):
    permute = np.random.permutation(len(x))
    for i in range(0, len(x)-batchsize, batchsize):
        indices = permute[i:i+batchsize]
        yield x[indices], y[indices]    


# In[ ]:


sess = tf.Session()

sess.run(tf.global_variables_initializer())

for epoch in range(2000 + 1):
    for batch_x, batch_t in batches(train_x, train_t, 32):
        sess.run(optimizer, {x: batch_x, t: batch_t})
    print(epoch, sess.run(loss, {x: test_x, t: test_t}))

test_p = sess.run(y, {x: test_x})


# ## Visualize results on test data

# In[ ]:


triples = zip(test_x[:10], test_t[:10], test_p[:10])
_, axis = plt.subplots(5, 3, figsize=(10,20))
plt.tight_layout(), plt.gray()
for (rgb, d, p), (ax1, ax2, ax3) in zip(triples, axis):
    ax1.axis('off'), ax2.axis('off'), ax3.axis('off')
    ax1.imshow(rgb)
    ax2.imshow(sp.misc.imresize(d, rgb.shape))
    ax3.imshow(sp.misc.imresize(p, rgb.shape))
plt.show()

