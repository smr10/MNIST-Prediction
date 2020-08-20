#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[3]:



print ("Train Images: ", mnist.train.images.shape)
print ("Train Labels  ", mnist.train.labels.shape)
print ("Test Images:  " , mnist.test.images.shape)
print ("Test Labels:  ", mnist.test.labels.shape)


# In[4]:


sess = tf.InteractiveSession()


# In[5]:


x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# In[6]:


# Weight tensor
W = tf.Variable(tf.zeros([784, 10],tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10],tf.float32))


# In[7]:


sess.run(tf.global_variables_initializer())


# In[8]:


tf.matmul(x,W) + b


# In[9]:


mnist


# In[10]:


y = tf.nn.softmax(tf.matmul(x,W) + b)


# In[11]:


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[12]:


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[13]:


for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# In[14]:


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )


# In[15]:


sess.close() #finish the session


# In[16]:


#Deep Learning applied on MNIST


# In[17]:


#In the first part, we learned how to use a simple ANN to classify MNIST. Now we are going to expand our knowledge using a Deep Neural Network.

#Architecture of our network is:

#(Input) -> [batch_size, 28, 28, 1] >> Apply 32 filter of [5x5]
#(Convolutional layer 1) -> [batch_size, 28, 28, 32]
#(ReLU 1) -> [?, 28, 28, 32]
#(Max pooling 1) -> [?, 14, 14, 32]
#(Convolutional layer 2) -> [?, 14, 14, 64]
#(ReLU 2) -> [?, 14, 14, 64]
#(Max pooling 2) -> [?, 7, 7, 64]
#[fully connected layer 3] -> [1x1024]
#[ReLU 3] -> [1x1024]
#[Drop out] -> [1x1024]
#[fully connected layer 4] -> [1x10]
#The next cells will explore this new architecture.


# In[18]:


import tensorflow as tf

# finish possible remaining session
sess.close()

#Start interactive session
sess = tf.InteractiveSession()


# In[19]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# - 55,000 data points
 # - mnist.train.images for inputs
 # - mnist.train.labels for outputs
    
#- 5,000 data points
 # - mnist.validation.images for inputs
  #- mnist.validation.labels for outputs

 #- 10,000 data points
  #- mnist.test.images for inputs
  #- mnist.test.labels for outputs
    


# In[20]:


width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem


# In[21]:


x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])


# In[22]:


x_image = tf.reshape(x, [-1,28,28,1])  
x_image


# In[23]:


W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs


# In[24]:


#convolution layer1
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1


# In[25]:


convolve1


# In[26]:


#relu (activation function)
h_conv1 = tf.nn.relu(convolve1)


# In[27]:


h_conv1


# In[28]:


conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv1


# In[29]:


#Convolutional Layer 2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs


# In[30]:


convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2


# In[31]:


h_conv2 = tf.nn.relu(convolve2)


# In[32]:


conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv2


# In[33]:


#Fully Connected Layer
layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])


# In[34]:


W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs


# In[35]:


fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1


# In[36]:


h_fc1 = tf.nn.relu(fcl)
h_fc1


# In[37]:


#Dropout Layer, Optional phase for reducing overfitting
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
layer_drop


# In[38]:


#Readout Layer (Softmax Layer)
#Type: Softmax, Fully Connected Layer
#Weights and Biases
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]


# In[39]:


#Matrix Multiplication (applying weights and biases)
fc=tf.matmul(layer_drop, W_fc2) + b_fc2


# In[40]:


#Apply the Softmax activation Function
y_CNN= tf.nn.softmax(fc)
y_CNN


# In[41]:


import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))


# In[42]:


#Loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))


# In[43]:


#Define the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[44]:


correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))
correct_prediction


# In[45]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy


# In[46]:


#Run session, train
sess.run(tf.global_variables_initializer())


# In[47]:


for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# In[48]:


# evaluate in batches to avoid out-of-memory issues
n_batches = mnist.test.images.shape[0] // 50
cumulative_accuracy = 0.0
for index in range(n_batches):
    batch = mnist.test.next_batch(50)
    cumulative_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print("test accuracy {}".format(cumulative_accuracy / n_batches))


# In[ ]:





# In[ ]:





# In[ ]:




