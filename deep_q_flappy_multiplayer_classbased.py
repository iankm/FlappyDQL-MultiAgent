#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import cv2
import sys
sys.path.append("games/")
import wrapped_flappy_bird as game
# from wrapped_flappy_bird import itercount
import random
import numpy as np
from collections import deque

GAME = 'flappybird_twocolor_horizontal' # the name of the game being played for log files
ACTIONS_PER_AGENT = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION_STEPS = 100000. # timesteps to observe before training
EXPLORATION_STEPS = 2100000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon 0.0001
INITIAL_EPSILON = 0.2 # starting value of epsilon 0.0001
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LOAD_CHECKPOINTS = True
OLD_CHECKPOINTS = False
NUM_PLAYERS = 2
num_steps_upon_load = 1960000

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS_PER_AGENT])
    b_fc2 = bias_variable([ACTIONS_PER_AGENT])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

class DeepQNet:
    def __init__(self, name, x_0, sess):
        ### global constants
        self.actions = ACTIONS_PER_AGENT
        self.sess = sess

        ### class variables
        s, readout, last_layer = createNetwork() # a CNN for each agent
        self.name = name # name of the player for debugging
        self.s = s # input observations
        self.readout = readout # output of Q-Net corresponds to Q values of each action
        self.last_layer = last_layer
        self.D = deque() # to store the action replays of the most recent state transitions
        self.epsilon = INITIAL_EPSILON - ((INITIAL_EPSILON - FINAL_EPSILON) * num_steps_upon_load/EXPLORATION_STEPS) 
        self.s_t = np.stack((x_0, x_0, x_0, x_0), axis=2) # "observation" = last 4 images
        self.t = num_steps_upon_load # time step
        
        ### define cost function for each agent
        self.a = tf.placeholder("float", [None, self.actions]) # actions
        self.y = tf.placeholder("float", [None]) # "true" rewards
        self.minibatch_actions = tf.reduce_sum(tf.mul(self.readout, self.a), reduction_indices=1) # predicted rewards for each action
        self.cost = tf.reduce_mean(tf.square(self.y - self.minibatch_actions))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

	print("initial epsilon: " + str(self.epsilon))
        
    def get_next_action(self):
        readout_t = self.readout.eval(feed_dict={self.s : [self.s_t]})[0]
        a_t = np.zeros([self.actions ])
        action_index = 0
        if self.t % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                # print('\t' + self.name + ': Random Action')
                action_index = random.randrange(self.actions)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing
        q_value = np.max(readout_t)
        return a_t, q_value 

    def get_consequences(self, r_t, x_t1, a_t, terminal):
        s_t1 = np.append(x_t1, self.s_t[:, :, :3], axis=2)

        if self.epsilon > FINAL_EPSILON and self.t > (num_steps_upon_load + OBSERVATION_STEPS):
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        self.D.append((self.s_t, a_t, r_t, s_t1, terminal))
        if len(self.D) > REPLAY_MEMORY:
            self.D.popleft()

        # only train if done observing
        if self.t > (OBSERVATION_STEPS + num_steps_upon_load):
	    # inserted only to resume training from saved checkpoint with correct epsilon...
	    #if self.t == OBSERVATION_STEPS + 1:
	    #	self.t = num_steps_upon_load
            #	self.epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * num_steps_upon_load/ EXPLORATION_STEPS 
            # sample a minibatch to train on
            minibatch = random.sample(self.D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = self.readout.eval(feed_dict = {self.s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            self.train_step.run(feed_dict = {
                self.y : y_batch,
                self.a : a_batch,
                self.s : s_j_batch}
            )

        # update the old values
        self.s_t = s_t1
        self.t += 1


def trainNetworks(sess):
    '''
        A framework to train one or many agents in a PyGame environment. In this instance we instantiate two agents, but you can change this; just make sure your game supports multiple players (accepts and returns an action and reward for each player)
    '''
    
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # printing
    # a_file = open("logs_" + GAME + "/readout.txt", 'w')
    # h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS_PER_AGENT)
    do_nothing[0] = 1
    x_0, _, _, terminal = game_state.frame_step(do_nothing, do_nothing)
    x_0 = cv2.cvtColor(cv2.resize(x_0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_0 = cv2.threshold(x_0,1,255,cv2.THRESH_BINARY)

    ### instantiate two q-learning agents
    q_learner1 = DeepQNet('player1', x_0, sess)
    q_learner2 = DeepQNet('player2', x_0, sess)

    ### saving future networks and loading pre-saved ones
    saver = tf.train.Saver() #### put tf.all_variables() as arg??????????
    sess.run(tf.initialize_all_variables())
    if(LOAD_CHECKPOINTS):
        if(OLD_CHECKPOINTS):
            checkpoint = tf.train.get_checkpoint_state("github_networks")
        else:
            checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
           saver.restore(sess, checkpoint.model_checkpoint_path)
           print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
           print("Error: Could not find old network weights")
	   exit()

    # start training
    t = num_steps_upon_load
    while "flappy bird" != "donald trump":
        # choose an action epsilon greedily
        itercount = game.itercount # ask how many sessions we've played

        ### ask what action each player chooses to do now
        a_t_1, q_t_1 = q_learner1.get_next_action()
        a_t_2, q_t_2 = q_learner2.get_next_action()

        ### run the selected action and observe next state and reward
        x_t1_colored, r_t_1, r_t_2, terminal = game_state.frame_step(a_t_1, a_t_2)
        # print (a_t_1, q_t_1, r_t_1, a_t_2, q_t_2, r_t_2, terminal)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        next_obs = np.reshape(x_t1, (80, 80, 1))

        ### incorporate environment's state and reward into each agents' losses
        q_learner1.get_consequences(r_t_1, next_obs, a_t_1, terminal)
        q_learner2.get_consequences(r_t_2, next_obs, a_t_2, terminal)
        
        t += 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print ('Frame Step: ' + str(t) + ' STATE: ' + str(state) \
                + ' Q_MAX1: ' + str(q_t_1) + ' Q_MAX2: ' + str(q_t_2) \
		+ ' eps_1: ' + str(q_learner1.epsilon) + ' eps_2: ' + str(q_learner2.epsilon))
        if t % 10000 == 0:
            print('saved networks! to saved_networks/' + GAME + '-multi_agent_dqn')
            saver.save(sess, 'saved_networks/' + GAME + '-multi_agent_dqn', global_step = t)

        # print info
        state = ""
        if t <= num_steps_upon_load + OBSERVATION_STEPS:
            state = "observe"
        elif t > num_steps_upon_load + OBSERVATION_STEPS and t <= num_steps_upon_load + OBSERVATION_STEPS + EXPLORATION_STEPS:
            state = "explore"
        else:
            state = "train"

        # if itercount % 50 == 0:
        #    print("iteration:", itercount, "/ STATE", state, "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[curr_obs]})[0]]) + '\n')
            cv2.imwrite("logcurr_obsetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    trainNetworks(sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
