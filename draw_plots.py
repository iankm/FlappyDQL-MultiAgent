from __future__ import division
import numpy as np
import re
#import matplotlib.pyplot as plt
import pylab as plt

n = 100 # moving average window

def moving_average(a, n=3):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

try:
	f = open('./multiplayer_training_log.txt', 'r')
except:
	print 'failed to open file'
	exit(1)

# match lines of form: "Crashed Player 1. Starting New Session. Iteration: 662, Score: 36"
# and "Frame Step: 900000 STATE: observe Q_MAX1: 4.02245 Q_MAX2: 3.07013"
scores, iterations = [], 0
frame_steps, qmaxs1, qmaxs2 = 0, [], []
for line in f:
	iter_num = re.findall("Starting New Session. Iteration: (\d+)", line)
	score = re.findall(", Score: (\d+)", line)
	if len(iter_num) > 0:
		iterations += 1
		scores.append(int(score[0]))

	qmax1 = re.findall("Q_MAX1: (\d+)", line)
	qmax2 = re.findall("Q_MAX2: (\d+)", line)
	if len(qmax1) == len(qmax2) == 1:
		frame_steps += 1
		qmaxs1.append(int(qmax1[0]))
		qmaxs2.append(int(qmax2[0]))

### plot scores as function of iteration
ma = moving_average(np.array(scores), n)
plt.plot(range(iterations), scores, 'b', label='Raw Scores')
plt.plot(range(iterations - n + 1), ma, 'r', label='100-Moving Average')
plt.legend(loc = 'upper left')
plt.xlabel("Number of Games Played")
plt.title("Score by the Terminal State of Each Iteration of Game-play")
plt.grid()
plt.show()

### plot q values of players 1 and 2 as function of frame step
n = 10
plt.figure(2)
ma_p1 = moving_average(np.array(qmaxs1), n)
ma_p2 = moving_average(np.array(qmaxs2), n)
plt.plot(range(frame_steps - n + 1), ma_p1, 'b', label='10-Moving Average of Player 1\'s Q_max')
plt.plot(range(frame_steps - n + 1), ma_p2, 'r', label='10-Moving Average of Player2\'s Q_max')
plt.xlabel("Number of Frame Steps (in 1000s)")
plt.legend(loc = 'upper left')
plt.title("Maximum Q-value For Each Player During Training")
plt.grid()
plt.show()
