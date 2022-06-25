from re import I, T
from typing import Counter
import numpy as np
from copy import copy
from timeit import timeit
from random import randint

from numpy.lib.polynomial import polysub

s0 = {
    1:' ', 2:' ', 3:' ',
    4:' ', 5:' ', 6:' ',
    7:' ', 8:' ', 9:' '
}
def reset_board(position):
    for key in position.keys():
        position[key] = ' '
    return position

def check_draw(position):
    for key in position.keys():
        if(position[key]==' '):
            return False
    return True
def heuristic(position):
    heuristic_table = {
        1:30, 2:20, 3:30,
        4:20, 5:40, 6:20,
        7:30, 8:20, 9:30
    }
    if(position[1]==position[2] and position[1]==position[3] and position[1]!=' '):
        if position[1] == 'x':
            return 10000
        elif position[1] == 'o':
            return -10000
    if(position[4]==position[5] and position[5]==position[6] and position[4]!=' '):
        if position[4] == 'x':
            return 10000
        elif position[4] == 'o':
            return -10000
    if(position[7]==position[8] and position[8]==position[9] and position[7]!=' '):
        if position[7] == 'x':
            return 10000
        elif position[7] == 'o':
            return -10000
    if(position[1]==position[4] and position[4]==position[7] and position[1]!=' '):
        if position[1] == 'x':
            return 10000
        elif position[1] == 'o':
            return -10000
    if(position[2]==position[5] and position[5]==position[8] and position[2]!=' '):
        if position[2] == 'x':
            return 10000
        elif position[2] == 'o':
            return -10000
    if(position[3]==position[6] and position[6]==position[9] and position[3]!=' '):
        if position[3] == 'x':
            return 10000
        elif position[3] == 'o':
            return -10000
    if(position[1]==position[5] and position[5]==position[9] and position[1]!=' '):
        if position[1] == 'x':
            return 10000
        elif position[1] == 'o':
            return -10000
    if(position[3]==position[5] and position[5]==position[7] and position[1]!=' '):
        if position[3] == 'x':
            return 10000
        elif position[3] == 'o':
            return -10000
    if check_draw(position):
        return 0
    heuristic_value = 0
    for key in position.keys():
        if position[key] == 'x':
            heuristic_value += heuristic_table[key]
        elif position[key] == 'o':
            heuristic_value -= heuristic_table[key]
    return heuristic_value


def minimax(position, depth, max, counter):
    position_value = heuristic(position)
    if position_value > 1000:
        return 10000, None, counter
    if position_value < -1000:
        return -10000, None, counter
    if check_draw(position):
        return 0, None, counter
    if depth==0:
        return position_value, None, counter
    if max:
        best_score = -100000
        for key in position.keys():
            if position[key]==' ':
                counter += 1
                position[key] = 'x'
                result = minimax(position, depth-1, False, counter)
                score = result[0]
                counter = result[2]
                position[key] = ' '
                if score > best_score:
                    best_score = score
                    best_key = key
        return best_score, best_key, counter
    else:
        best_score = 100000
        for key in position.keys():
            if position[key]==' ':
                counter += 1
                position[key] = 'o'
                result = minimax(position, depth-1, True, counter)
                score = result[0]
                counter = result[2]
                position[key] = ' '
                if score < best_score:
                    best_score = score
                    best_key = key
        return best_score, best_key, counter

def minimax_wtih_ab(position, depth, alpha, beta, max_player, counter):
    position_value = heuristic(position)
    if position_value > 1000:
        return 10000, None, counter
    if position_value < -1000:
        return -10000, None, counter
    if check_draw(position):
        return 0, None, counter
    if depth==0:
        return position_value, None, counter
    if max_player:
        best_score = -100000
        for key in position.keys():
            if position[key]==' ':
                counter += 1
                position[key] = 'x'
                result = minimax_wtih_ab(position, depth-1, alpha, beta, False, counter)
                score = result[0]
                counter = result[2]
                position[key] = ' '
                if score > best_score:
                    best_score = score
                    best_key = key
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
        return best_score, best_key, counter
    else:
        best_score = 100000
        for key in position.keys():
            if position[key]==' ':
                counter += 1
                position[key] = 'o'
                result = minimax_wtih_ab(position, depth-1, alpha, beta, True, counter)
                score = result[0]
                counter = result[2]
                position[key] = ' '
                if score < best_score:
                    best_score = score
                    best_key = key
                beta = min(beta, score)
                if beta <= alpha:
                    break
        return best_score, best_key, counter

def AI(position, D, ab, random, max_player, counter):
    if check_draw(position):
        return position, counter
    if random or D==0:
        while True:
            key = randint(1, 9)
            if position[key] == ' ':
                if max_player:
                    position[key] = 'x'
                    return position, counter
                else:
                    position[key] = 'o'
                    return position, counter
            else:
                continue
    if ab:
        result = minimax_wtih_ab(position, D, alpha=-100000, beta=100000, max_player=max_player,counter=0)
        key = result[1]
        # print(key)
        counter = result[2]
        if max_player:
            position[key] = 'x'
            return position, counter
        else:
            position[key] = 'o'
            return position, counter
    result = minimax(position, D, max_player, counter)
    key = result[1]
    # print(key)
    counter = result[2]
    if max_player:
        position[key] = 'x'
        return position, counter
    else:
        position[key] = 'o'
        return position, counter

def AIvsAI(position, D1, ab1, rand1, D2, ab2, rand2, player, counter1, counter2):
    while not check_draw(position):
        if player:
            result = AI(position, D1, ab1, rand1, player, 0)
            position = result[0]
            counter1 += result[1]
        else:
            result = AI(position, D2, ab2, rand2, player, 0)
            position = result[0]
            counter2 += result[1]
        position_value = heuristic(position)
        if position_value > 1000:
            print('x won(10000)')
            reset_board(position)
            return position_value, position, counter1, counter2
        elif position_value < -1000:
            print('o won(-10000)')
            reset_board(position)
            return position_value, position, counter1, counter2
        player = not player
    print('draw(0)')
    reset_board(position)
    return 0, position, counter1, counter2
# AI vs Random_AI
round = 1
x_wins = 0
x_loses = 0
draws = 0
while round <= 1000:
    result = AIvsAI(s0, 9, True, False, 0, False, True, True, 0, 0)[0]
    if result == 10000:
        x_wins += 1
    elif result == -10000:
        x_loses += 1
    elif result == 0:
        draws += 1
    round += 1
print(f'x_wins: {x_wins}')
print(f'x_loses: {x_loses}')
print(f'draws: {draws}')
