# Student agent: Add your own agent here
import copy
import random
from collections import defaultdict

import numpy as np

from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        class MonteCarloTree():
            def __init__(self, cur_state, board_size, parent=None, parent_action=None):
                self.cur_state = cur_state
                self.parent = parent
                self.parent_action = parent_action
                self.board_size = board_size
                self.children = []
                self._number_of_visits = 0
                self._results = defaultdict(int)
                self._results[1] = 0
                self._results[-1] = 0
                self._untried_actions = None
                self._untried_actions = self.untried_actions()
                self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
                self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
                return

            def untried_actions(self):
                self._untried_actions = self.cur_state.get_actions()
                return self._untried_actions

            def q(self):
                wins = self._results[1]
                loses = self._results[-1]
                return wins - loses

            def n(self):
                return self._number_of_visits

            def expand(self):
                action = self._untried_actions.pop()
                next_state = self.move(action)
                child_node = MonteCarloTree(next_state, self.board_size, parent=self, parent_action=action)
                return child_node

            def is_terminal_node(self):
                return self.game_over(self.cur_state)

            def game_over(self, cur_state):
                father = dict()
                for r in range(self.board_size):
                    for c in range(self.board_size):
                        father[(r, c)] = (r, c)

                def find(pos):
                    if father[pos] != pos:
                        father[pos] = find(father[pos])
                    return father[pos]

                def union(pos1, pos2):
                    father[pos1] = pos2

                for r in range(self.board_size):
                    for c in range(self.board_size):
                        for dir, move in enumerate(
                                self.moves[1:3]
                        ):  # Only check down and right
                            if self.cur_state[2][r, c, dir + 1]:
                                continue
                            pos_a = find((r, c))
                            pos_b = find((r + move[0], c + move[1]))
                            if pos_a != pos_b:
                                union(pos_a, pos_b)

                for r in range(self.board_size):
                    for c in range(self.board_size):
                        find((r, c))
                p0_r = find(tuple(cur_state[0]))
                p1_r = find(tuple(cur_state[1]))
                p0_score = list(father.values()).count(p0_r)
                p1_score = list(father.values()).count(p1_r)
                if p0_r == p1_r:
                    return False, p0_score, p1_score
                player_win = None
                win_blocks = -1
                if p0_score > p1_score:
                    player_win = 0
                    win_blocks = p0_score
                elif p0_score < p1_score:
                    player_win = 1
                    win_blocks = p1_score
                else:
                    player_win = -1  # Tie

                return True, p0_score, p1_score

            def move(self, action):
                #create a new board
                next_pos, dir = action
                r, c = next_pos
                new_board = self.set_barrier(r, c, dir)
                return next_pos,self.cur_state[1], new_board

            def set_barrier(self,r,c,dir):
                # Set the barrier to True
                chess_board = copy.deepcopy(self.cur_state[2])
                chess_board[r, c, dir] = True
                # Set the opposite barrier to True
                move = self.moves[dir]
                chess_board[r + move[0], c + move[1], self.opposites[dir]] = True
                return chess_board

            def get_actions(self):
                actions = []

                for i in range(50):
                    ori_pos = copy.deepcopy(self.cur_state[0])
                    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
                    steps = np.random.randint(0, (self.board_size + 1) // 2 + 1)

                    # Random Walk
                    for _ in range(steps):
                        r, c = self.cur_state[0]
                        dir = np.random.randint(0, 4)
                        m_r, m_c = moves[dir]
                        ori_pos = (r + m_r, c + m_c)

                        # Special Case enclosed by Adversary
                        k = 0
                        while self.cur_state[2][r, c, dir] or ori_pos == self.cur_state[1]:
                            k += 1
                            if k > 300:
                                break
                            dir = np.random.randint(0, 4)
                            m_r, m_c = moves[dir]
                            ori_pos = (r + m_r, c + m_c)

                        if k > 300:
                            ori_pos = self.cur_state[0]
                            break

                    # Put Barrier
                    dir = np.random.randint(0, 4)
                    r, c = ori_pos
                    while self.cur_state[2][r, c, dir]:
                        dir = np.random.randint(0, 4)
                    action = (ori_pos,dir)
                    actions.append(action)
                return actions

            def simulate(self):
                cur_state = self.cur_state

                while not self.game_over(cur_state):
                    moves = self.get_actions()
                    step = random.randint(0, len(moves))
                    move = moves[step]
                    cur_state = self.move(move)
                return self.get_result()

            def backTracking(self,result):
                self._number_of_visits += 1.
                self._results[result] += 1.
                if self.parent:
                    self.parent.backpropagate(result)



    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return
        return my_pos, self.dir_map["u"]
