# To Do: Change reward to -100
# in frame_step

import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 10000
SCREENWIDTH  = 288
SCREENHEIGHT = 512
DISPLAY = False

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 125 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79
itercount = 0

PLAYER_WIDTH = IMAGES['player1'][0].get_width()
PLAYER_HEIGHT = IMAGES['player1'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.player1x = int(SCREENWIDTH * 0.25)
        self.player1y = int((SCREENHEIGHT - PLAYER_HEIGHT - 5) / 2)
        self.player2x = int(SCREENWIDTH * 0.1)
        self.player2y = int((SCREENHEIGHT + PLAYER_HEIGHT + 5) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.player1VelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.player2VelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.player1Flapped = False # True when player flaps
        self.player2Flapped = False # True when player flaps

    def frame_step(self, input_actions1, input_actions2):
        global itercount
        pygame.event.pump()

        reward_1 = 0.1
        reward_2 = 0.1
        terminal = False

        if sum(input_actions1) != 1:
            raise ValueError('Multiple input actions!')
        if sum(input_actions2) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions1[0] == 1: do nothing 1
        # input_actions1[1] == 1: flap bird 1
        # input_actions2[0] == 1: do nothing 2
	    # input_actions2[1] == 1: flap bird 2
        if input_actions1[1] == 1:
            if self.player1y > -2 * PLAYER_HEIGHT:
                self.player1VelY = self.playerFlapAcc
                self.player1Flapped = True
                #SOUNDS['wing'].play()
        if input_actions2[1] == 1:
            if self.player2y > -2 * PLAYER_HEIGHT:
                self.player2VelY = self.playerFlapAcc
                self.player2Flapped = True
                #SOUNDS['wing'].play()

        # check for score
        playerMidPos1 = self.player1x + PLAYER_WIDTH / 2
        playerMidPos2 = self.player2x + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos1 < pipeMidPos + 4:
                self.score += 1
                #SOUNDS['point'].play()
                reward_1 = 1
            if pipeMidPos <= playerMidPos2 < pipeMidPos + 4:
                self.score += 1
                reward_2 = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # players' movements
        if self.player1VelY < self.playerMaxVelY and not self.player1Flapped:
            self.player1VelY += self.playerAccY
        if self.player1Flapped:
            self.player1Flapped = False
        self.player1y += min(self.player1VelY, BASEY - self.player1y - PLAYER_HEIGHT)
        if self.player1y < 0:
            self.player1y = 0


        if self.player2VelY < self.playerMaxVelY and not self.player2Flapped:
            self.player2VelY += self.playerAccY
        if self.player2Flapped:
            self.player2Flapped = False
        self.player2y += min(self.player2VelY, BASEY - self.player2y - PLAYER_HEIGHT)
        if self.player2y < 0:
            self.player2y = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash= checkCrash({'x': self.player1x, 'y': self.player1y,
                                'index': self.playerIndex},
			                {'x': self.player2x, 'y': self.player2y,
			                     'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            itercount += 1
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            terminal = True
            print "Crashed. Starting New Session. Iteration: " + str(itercount) + ", Score: " + str(self.score)
            self.__init__()
            reward_1 = -1
            reward_2 = -1

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player1'][self.playerIndex],
                    (self.player1x, self.player1y))
        SCREEN.blit(IMAGES['player2'][self.playerIndex],
                    (self.player2x, self.player2y))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        if(DISPLAY):
            pygame.display.update()
        FPSCLOCK.tick(FPS)
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        return image_data, reward_1, reward_2, terminal

def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player1, player2, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player1['index']
    player1['w'] = IMAGES['player1'][0].get_width()
    player1['h'] = IMAGES['player1'][0].get_height()
    player2['w'] = IMAGES['player2'][0].get_width()
    player2['h'] = IMAGES['player2'][0].get_height()

    # if player crashes into ground
    if player1['y'] + player1['h'] >= BASEY - 1 or player2['y'] + player2['h'] >= BASEY - 1:
        return True
    else:

        player1Rect = pygame.Rect(player1['x'], player1['y'],
                      player1['w'], player1['h'])
        player2Rect = pygame.Rect(player2['x'], player2['y'],
                      player2['w'], player2['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            p1HitMask = HITMASKS['player1'][pi]
            p2HitMask = HITMASKS['player2'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if either bird collided with upipe or lpipe or other bird
            uCollide = pixelCollision(player1Rect, uPipeRect, p1HitMask, uHitmask) or pixelCollision(player2Rect, uPipeRect, p2HitMask, uHitmask)
            lCollide = pixelCollision(player1Rect, lPipeRect, p1HitMask, lHitmask) or pixelCollision(player2Rect, lPipeRect, p2HitMask, lHitmask)
            playerCollide = pixelCollision(player1Rect, player2Rect, p1HitMask, p2HitMask)

            if uCollide or lCollide or playerCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
