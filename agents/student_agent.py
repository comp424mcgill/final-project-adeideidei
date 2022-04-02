# Student agent: Add your own agent here
import copy
import random
from collections import defaultdict
from typing import List

import numpy as np
import logging

from agents.agent import Agent
from store import register_agent


class MonteCarloTree:
    def __init__(self, cur_state, board_size, max_step, dir, parent=None, parent_action=None):

        logging.info(
            "-----------------init a new tree --------------------------------------"
        )
        self.cur_state = cur_state
        self._number_of_visits = 1
        self.parent = parent
        self.dir = dir
        self.max_step = max_step
        self.parent_action = parent_action
        self.board_size = board_size
        self.children = []
        self._number_of_blocks = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        return

    def get_state(self):
        return self.cur_state

    def get_dir(self):
        return self.dir

    def untried_actions(self):
        self._untried_actions = self.get_actions(self.cur_state)
        return self._untried_actions

    def win_rate(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def get_number_visited(self):
        return self._number_of_visits

    def get_win_blocks(self):
        return self._number_of_blocks

    def expand(self):
        logging.info(
            "-----------------expand tree --------------------------------------"
        )
        action = self._untried_actions.pop()
        old_board = copy.deepcopy(self.cur_state[2])
        next_state = self.move(action, old_board)
        child_node = MonteCarloTree(next_state, self.board_size, self.max_step, action[1], parent=self,
                                    parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.game_over(self.cur_state)

    def game_over(self, cur_state):
        """"
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

        p0_r = find(cur_state[0])
        p1_r = find(cur_state[1])
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)

        if p0_r == p1_r:
            return False, p0_score, p1_score
        """
        return True, 0, 0

    def move(self, action, oldboard):
        # create a new board
        next_pos, dir = action
        r, c = next_pos
        new_board = self.set_barrier(r, c, dir, oldboard)
        return next_pos, self.cur_state[1], new_board

    def set_barrier(self, r, c, dir, old_board):

        # Set the barrier to True
        chess_board = copy.deepcopy(old_board)
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True
        return chess_board

    def get_actions(self, cur_state):
        logging.info(
            "-----------------start getting all the possible moves--------------------------------------"
        )
        actions = []

        for i in range(5):
            logging.info(
                "-----------------start getting all the possible moves + 1--------------------------------------"
            )

            ori_pos = copy.deepcopy(cur_state[0])
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            steps = np.random.randint(0, self.max_step)

            # Random Walk
            for _ in range(steps):
                r, c = cur_state[0]
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                ori_pos = (r + m_r, c + m_c)

                # Special Case enclosed by Adversary
                k = 0
                while cur_state[2][r, c, dir] or ori_pos == cur_state[1]:
                    k += 1
                    if k > 300:
                        break
                    dir = np.random.randint(0, 4)
                    m_r, m_c = moves[dir]
                    ori_pos = (r + m_r, c + m_c)

                if k > 300:
                    ori_pos = cur_state[0]
                    break

            # Put Barrier
            dir = np.random.randint(0, 4)
            r, c = ori_pos
            while cur_state[2][r, c, dir]:
                dir = np.random.randint(0, 4)
            action = (ori_pos, dir)
            actions.append(action)
        return actions

    def simulate(self):
        logging.info(
            "-----------------start simulating the result--------------------------------------"
        )
        cur_state = self.cur_state

        while not self.game_over(cur_state)[0]:
            logging.info(
                "-----------------simulating--------------------------------------"
            )
            moves = self.get_actions(cur_state)
            step = random.randint(0, len(moves) - 1)
            move = moves[step]
            cur_state = self.move(move, cur_state[2])
            if self.game_over(cur_state)[0]:
                break

            cur_state = self.opp_move(cur_state)
        logging.info(
            "-----------------finished simulating the result--------------------------------------"
        )
        return self.game_over(cur_state)

    def opp_move(self, cur_state):
        logging.info(
            "-----------------simulate opp moves--------------------------------------"
        )
        opp_pos = copy.deepcopy(cur_state[1])
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, self.max_step)

        # Random Walk
        for _ in range(steps):
            logging.info(
                "-----------------simulate opp moves + 1--------------------------------------"
            )
            r, c = opp_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            opp_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while cur_state[2][r, c, dir] or opp_pos == cur_state[0]:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                opp_pos = (r + m_r, c + m_c)

            if k > 300:
                opp_pos = cur_state[1]
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = opp_pos
        while cur_state[2][r, c, dir]:
            dir = np.random.randint(0, 4)

        r, c = opp_pos
        new_board = self.set_barrier(r, c, dir, cur_state[2])
        return cur_state[0], opp_pos, new_board

    def backtracking(self, result):
        self._number_of_blocks += (result[1] - result[2])
        self._number_of_visits += 1
        if result[0]:
            self._results[0] += 1
        else:
            self._results[1] += 1
        if self.parent is not None:
            self.parent.backpropagate(result)

    def best_node(self):
        max = []
        for c in self.children:
            value = (c.win_rate() / c.get_number_visited()) + 0.1 * c.get_win_blocks()
            value = value + 0.1 * np.sqrt(2 * np.log(self.get_number_visited() / c.get_number_visited()))
            max.append(value)
        return self.children[np.argmax(max)]

    def select(self, m):
        logging.info(
            "-----------------select chdilren--------------------------------------"
        )
        cur_n = self

        while not cur_n.game_over(cur_n.cur_state)[0]:
            if m < 10:
                logging.info(
                    "-----------------select chdilren part 1--------------------------------------"
                )
                m += 1
                return cur_n.expand()

            cur_n = cur_n.best_node()
        return cur_n

    def pick_children(self):
        logging.info(
            "-----------------pick children --------------------------------------"
        )
        for i in range(15):
            cur_node = self.select(i)
            result = cur_node.simulate()
            cur_node.backtracking(result)
        return self.best_node()



























"""
========================================================= NEW =========================================================
"""


class Action:
    """
    A class to store a step by storing its new and old positions in [x, y] and the new barrier to put
    """

    def __init__(self, start_pos: tuple, end_pos: tuple, barrier_dir: int):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.barrier_dir = barrier_dir
        self.step_taken = -1
        self.score = 0
    def set_score(self, score):
        self.score = score

    def set_barrier(self, r, c, dir, old_board: np.ndarray):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to True
        result = copy.deepcopy(old_board)
        result[r, c, dir] = True
        # Set the opposite barrier to True
        move = moves[dir]
        result[r + move[0], c + move[1], opposites[dir]] = True
        return result


    def game_finished(self, chess_board, my_pos, adv_pos, board_size : int):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))

        p0_r = find(my_pos)
        p1_r = find(adv_pos)
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)

        if p0_r == p1_r:
            return False, p0_score, p1_score

        return True, p0_score, p1_score





def heuristic(chess_board: np.ndarray, my_pos: tuple, adv_pos: tuple, max_step: int, actions: List[Action]) -> List[Action]:
    """
    Do heuristic here

    Parameters
    ----------
    chess_board: np.ndarray
        A numpy array of shape (x_max, y_max, 4)

    my_pos: tuple
        The position of the agent

    adv_pos: tuple
        The position of the adversary

    max_step: int
        The maximum step that can move

    actions: List[Action]
        The valid actions needed to be processed

    Returns
    -------
    best_step: Action
        The best chosen step from heuristic
    """

    # best_step = Action(my_pos, np.ndarray(my_pos), 0)
    max_index = 0
    max_score = -2000
    i = 0
    board_size, _, _ = chess_board.shape
    mid = (int(board_size/2), int(board_size/2))
    top_actions = []


    for i in range(0,len(actions)):
        action = actions[i]

       # print(i, "\n")
        score = 0
        cur_pos = action.end_pos
        pre_pos = action.start_pos
        #check if the player stay at the same place:
        #if cur_pos[0] == pre_pos[0] and  cur_pos[1] == pre_pos[1]:
            #score -= 20

        # check if the pos is further away from the middle
        distance_cur_mid = abs(cur_pos[0] - mid[0]) + abs(cur_pos[1] - mid[1])
        distance_pre_mid = abs(pre_pos[0] - mid[0]) + abs(pre_pos[1] - mid[1])
        if distance_cur_mid < distance_pre_mid:
            score += 20
        else:
            score -=20


        # check if the pos is further away from the adv pos compared to previous pos
        distance_cur = abs(cur_pos[0] - adv_pos[0]) + abs(cur_pos[1] - adv_pos[1])
        distance_pre = abs(pre_pos[0] - adv_pos[0]) + abs(pre_pos[1] - adv_pos[1])
        if distance_cur < max_step + 1:
            score -= 20
        else:
            score += 20

        if cur_pos[0] - adv_pos[0] > 0 and action.barrier_dir == 0:
            score += 40
        elif cur_pos[0] - adv_pos[0] < 0 and action.barrier_dir == 2:
            score += 40

        if cur_pos[1] - adv_pos[1] > 0 and action.barrier_dir == 3:
            score += 40
        elif cur_pos[1] - adv_pos[1] < 0 and action.barrier_dir == 1:
            score += 40


        #check if the place entered already has 2 walls
        numbers_border = 0;
        for j in range (0, 4):
            if chess_board[cur_pos[0], cur_pos[1], j]:
                numbers_border += 1
        if numbers_border >= 2:
            score -= 200
        else:
            score += 50

        #check if it can finished the game directly
        new_chess_board = action.set_barrier(cur_pos[0], cur_pos[1], action.barrier_dir, chess_board)
        board_size, _, _ = new_chess_board.shape
        game_result = action.game_finished(new_chess_board, cur_pos, adv_pos, board_size)

        if game_result[0] and game_result[1] > game_result[2]:

            score += 5000
        elif game_result[0] and game_result[1] <= game_result[2]:
            score -= 5000




        if score > max_score:
            max_score = score
            max_index = i
        action.set_score(score)
        if i <= 2:
            top_actions.append(action)
        else:
            for i in range(0, len(top_actions)):
                if top_actions[i].score < score:
                    top_actions[i] = action
        i += 1



    return top_actions


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def best_opp(self, chess_board: np.ndarray, my_pos: tuple, adv_pos: tuple, max_step: int, actions: List[Action]) -> Action:

        # best_step = Action(my_pos, np.ndarray(my_pos), 0)
        max_index = 0
        max_score = -2000
        i = 0
        board_size, _, _ = chess_board.shape
        mid = (int(board_size / 2), int(board_size / 2))

        for i in range(0, len(actions)):
            action = actions[i]

            # print(i, "\n")
            score = 0
            cur_pos = action.end_pos
            pre_pos = action.start_pos
            # check if the player stay at the same place:
            if cur_pos[0] == pre_pos[0] and  cur_pos[1] == pre_pos[1]:
                score -= 40

            # check if the pos is further away from the middle
            distance_cur_mid = abs(cur_pos[0] - mid[0]) + abs(cur_pos[1] - mid[1])
            distance_pre_mid = abs(pre_pos[0] - mid[0]) + abs(pre_pos[1] - mid[1])
            if distance_cur_mid < distance_pre_mid:
                score += 40
            else:
                score -= 40

            if cur_pos[0] - adv_pos[0] > 0 and action.barrier_dir == 0:
                score += 40
            elif cur_pos[0] - adv_pos[0] < 0 and action.barrier_dir == 2:
                score += 40

            if cur_pos[1] - adv_pos[1] > 0 and action.barrier_dir == 3:
                score += 40
            elif cur_pos[1] - adv_pos[1] < 0 and action.barrier_dir == 1:
                score += 40

            # check if the pos is further away from the adv pos compared to previous pos
            distance_cur = abs(cur_pos[0] - adv_pos[0]) + abs(cur_pos[1] - adv_pos[1])
            distance_pre = abs(pre_pos[0] - adv_pos[0]) + abs(pre_pos[1] - adv_pos[1])
            if distance_cur < max_step + 1:
                score -= 20
            else:
                score += 20

            # check if the place entered already has 2 walls
            numbers_border = 0;
            for j in range(0, 4):
                if chess_board[cur_pos[0], cur_pos[1], j]:
                    numbers_border += 1
            if numbers_border >= 2:
                score -= 200
            else:
                score += 50

            # check if it can finished the game directly
            new_chess_board = action.set_barrier(cur_pos[0], cur_pos[1], action.barrier_dir, chess_board)
            board_size, _, _ = new_chess_board.shape
            game_result = action.game_finished(new_chess_board, cur_pos, adv_pos, board_size)

            if game_result[0] and game_result[1] > game_result[2]:

                score += 5000

            elif game_result[0] and game_result[1] <= game_result[2]:

                score -= 5000

            if score > max_score:
                max_score = score
                max_index = i
            action.set_score(score)

            i += 1

        return actions[max_index]

    def check_valid_step(self, chess_board: np.ndarray, action: Action, adv_pos: tuple, max_step: int) -> bool:
        """
        Check if this new step is valid or not. If action is valid, update the step taken for this action.

        Parameters
        ----------
        chess_board: np.ndarray
            A numpy array of shape (x_max, y_max, 4)

        action: Action
            The action to do (move and put barrier)

        adv_pos: tuple
            The position of the adversary

        max_step: int
            The maximum step that can move

        Returns
        -------
        is_valid: bool
            If valid, return True; otherwise, return False
        """

        # Endpoint already has barrier or is boarder
        x, y = action.end_pos

        if chess_board[x, y, action.barrier_dir]:
            return False

        if np.array_equal(action.start_pos, action.end_pos):
            return True

        # BFS
        state_queue = [(action.start_pos, 0)]
        visited = {tuple(action.start_pos)}
        is_valid = False

        while state_queue and not is_valid:
            cur_pos, cur_step = state_queue.pop(0)

            # logging.info(cur_pos)

            x, y = cur_pos

            if cur_step == max_step:
                break

            for direction, move in enumerate(self.moves):
                if chess_board[x, y, direction]:
                    continue

                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])

                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue

                if np.array_equal(next_pos, action.end_pos):
                    action.step_taken = cur_step
                    is_valid = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_valid

    def get_valid_steps(self, chess_board: np.ndarray, my_pos: tuple, adv_pos: tuple, max_step: int) -> List[Action]:
        """
        Get all the valid steps that can be acted

        Parameters
        ----------
        chess_board: np.ndarray
            A numpy array of shape (x_max, y_max, 4)

        my_pos: tuple
            The position of the agent

        adv_pos: tuple
            The position of the adversary

        max_step: int
            The maximum step that can move

        Returns
        -------
        valid_actions: List[Action]
            All the valid actions
        """

        valid_actions = []
        board_size, _, _ = chess_board.shape
        x, y = my_pos

        for i in range(x - max_step, x + max_step + 1):
            if i < 0 or i >= board_size:
                continue

            for j in range(y - max_step, y + max_step + 1):
                if j < 0 or j >= board_size:
                    continue

                for k in range(0, 4):
                    # np.ndarray((2,), buffer=np.array([i, j]), dtype=int)
                    cur_action = Action(my_pos, (i, j), k)

                    # logging.info("i, j, k: %d, %d, %d", i, j, k)
                    # logging.info(np.ndarray((2,), buffer=np.array([i, j]), dtype=int))

                    if self.check_valid_step(chess_board, cur_action, adv_pos, max_step):
                        valid_actions.append(cur_action)


        return valid_actions

    def simulate(self, actions: List[Action], adv_pos, chessboard, max_step) -> Action :

        max_score = -10000
        i = 0
        max_index = 0

        for action in actions:
            #get the new chessboard based on the action
            new_chessboard = action.set_barrier(action.end_pos[0], action.end_pos[1], action.barrier_dir,chessboard)
            #get all the possible opp move
            opp_actions = self.get_valid_steps(new_chessboard, adv_pos, action.end_pos, max_step)
            #choose the best opp move
            opp_best_action = self.best_opp(new_chessboard, adv_pos, action.end_pos, max_step, opp_actions)
            #update a new chess board
            updated_chessboard = action.set_barrier(opp_best_action.end_pos[0],  opp_best_action.end_pos[1],  opp_best_action.barrier_dir, new_chessboard)
            #get all the new possible action for student
            new_actions_for_student = self.get_valid_steps(updated_chessboard, action.end_pos, opp_best_action.end_pos, max_step)
            new_best_action = self.best_opp(updated_chessboard, action.end_pos, opp_best_action.end_pos, max_step, new_actions_for_student)
            current_score = new_best_action.score + action.score
            #print(current_score)
            if current_score > max_score:
                max_index = i
            i += 1





        return actions[max_index]


    def step(self, chess_board: np.ndarray, my_pos: tuple, adv_pos: tuple, max_step: int):
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
        # MCT
        # cur_state = (my_pos, adv_pos, chess_board)
        # tree = MonteCarloTree(cur_state, chess_board.shape[0], max_step, 1)
        # best_choice = tree.pick_children()
        # return best_choice.get_state()[0], best_choice.get_dir

        # Do Heuristic
        actions = self.get_valid_steps(chess_board, my_pos, adv_pos, max_step)
       # print('length of this: %d', len(actions), "\n")
        best_steps = heuristic(chess_board, my_pos, adv_pos, max_step, actions)
        best_step = self.simulate(best_steps,adv_pos, chess_board, max_step)



        # print(actions, "\n")

        return best_step.end_pos, best_step.barrier_dir

        # dummy return
        # return my_pos, 0
        # python simulator.py --player_1 random_agent --player_2 student_agent --autoplay  --autoplay_runs 1000
        #  python simulator.py --player_1 random_agent --player_2 student_agent --display