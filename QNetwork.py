import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(x):
    return np.identity(16)[x:x+1]
env = gym.make("FrozenLake-v0")

# Input and Output size based on the env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

## PLACEHOLDER DOESNT EXIST IN TF v2
# These lines establish the feed-forward part of the network used to those actions
X = tf.placeholder(shape=[1,input_size], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size],0,0.01))
Qpred = tf.matmul(X,W)
Y = tf.placeholder(shape=[1,output_size],dtype = tf.float32)

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimzer(learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
dis= 0.99
num_episodes = 2000
# Create lists to contain total rewards and steps per episode
rList = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset the environment and get first new observation
        s = env.reset()
        e = 1./((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []

        # The Q-network training
        while not done:
            # Choose an action by greedily (with e chance of random action) from the Q-network
            Qs = sess.run(Qpred,feed_dict = {X: one_hot(s)})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            # Get new state and reward from environment
            s1, reward, done, _ = env.step(a)
            if done:
                # Update Q, and no Qs+1, since it's terminal state
                Qs[0,a] = reward
            else:
                # Obtain the Q_s1 values by feeding the new state through our network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
                # Update Q
                Qs[0,a] = reward + dis*np.max(Qs1)

            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: one_hot(s),Y:Qs})

            rAll += reward
print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
plt.bar(range(len(rList)),rList, color='blue')
plt.show()