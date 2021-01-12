import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

# Input and Output size based on the env
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.1

## PLACEHOLDER DOESNT EXIST IN TF v2
# These lines establish the feed-forward part of the network used to those actions
X = tf.placeholder(tf.float32, [None, input_size], name="input_x")
W1 = tf.get_variable("W1", shape=[input_size, output_size], initializer = tf.contrib.layers.xavier_initializer())
Qpred = tf.matmul(X,W1)

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(shape=[None,output_size],dtype = tf.float32)

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
dis= 0.99
num_episodes = 2000
# Create lists to contain total rewards and steps per episode
rList = []
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)



for i in range(num_episodes):
    step_count = 0
    # Reset the environment and get first new observation
    s = env.reset()
    e = 1./((i / 50) + 10)
    rAll = 0
    done = False

    # The Q-network training
    while not done:
        step_count += 1
        x = np.reshape(s,[1,input_size])
        # Choose an action by greedily (with e chance of random action) from the Q-network
        Qs = sess.run(Qpred,feed_dict = {X: x})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        # Get new state and reward from environment
        s1, reward, done, _ = env.step(a)
        if done:
            # Update Q, and no Qs+1, since it's terminal state
            Qs[0,a] = -100
        else:
            x1 = np.reshape(s1,[1,input_size])
            # Obtain the Q_s1 values by feeding the new state through our network
            Qs1 = sess.run(Qpred, feed_dict={X: x1})
            # Update Q
            Qs[0,a] = reward + dis*np.max(Qs1)

        # Train our network using target and predicted Q (Qpred) values
        sess.run(train, feed_dict={X: x,Y:Qs})

        s = s1

    rList.append(step_count)
    print("Episode: {} steps: {}".format(i,step_count))
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break
# See our trained network in action
observation = env.reset()
reward_sum = 0
while True:
    env.render()
    x = np.reshape(observation, [1,input_size])
    Qs = sess.run(Qpred, feed_dict={X:x})
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break
# print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
# plt.bar(range(len(rList)),rList, color='blue')
# plt.show()