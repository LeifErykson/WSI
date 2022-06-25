import numpy as np
from matplotlib import pyplot as plt


class Q_learning:
    def __init__(self, path):
        self.path = path
        self.current_y = 0
        self.current_x = 0
        self.maze = None
        self.rewards = None
        self.width = 0
        self.height = 0
        self.epsilon = 0
        self.q_table = None
        self.next_action = 0
        self.shortest_path = []
        self.discount_factor = 0
        self.learning_rate = 0
        self.actions = ['up', 'right', 'down', 'f']
        self.training_moves = []
        self.training_rewards = []
        self.starting_x = 0
        self.starting_y = 0
    def from_file_to_array_maze(self):
        with open(self.path) as f:
            lines = f.readlines()
            rows = []
            axis = 0
            for line in lines:
                row = np.array([])
                for sign in line:
                    if sign == '\n':
                        axis += 1
                        continue
                    row = np.append(row, sign)
                rows.append(row)
            maze_array = np.array(rows)
            self.maze = maze_array
            self.height, self.width = np.shape(maze_array)
            self.q_table = np.zeros((self.height, self.width, 4))


    def maze_rewards(self):
        maze_shape = np.shape(self.maze)
        rewards = np.full(shape = maze_shape, fill_value = -1)
        y = 0
        for row in self.maze:
            x = 0
            for sign in row:
                if sign == "F":
                    rewards[y][x] = 100
                elif sign == "#":
                    rewards[y][x] = -100
                elif sign == "S":
                    self.starting_x = x
                    self.starting_y = y
                x += 1
            y += 1
        self.rewards = rewards


    def is_terminal_state(self):
        if self.rewards[self.current_y, self.current_x] == -1:
            return False
        else:
            return True


    def get_starting_location(self):
        self.current_y = np.random.randint(self.height)
        self.current_x = np.random.randint(self.width)
        if self.is_terminal_state():
            self.get_starting_location()


    def get_next_action(self, epsilon):
        if np.random.random() < epsilon:
            self.next_action= np.argmax(self.q_table[self.current_y, self.current_x])
        else:
            self.next_action =  np.random.randint(4)


    def next_locaction(self):
        if self.next_action == 0 and self.current_y > 0:
            self.current_y -= 1
        elif self.next_action == 1 and self.current_x < self.width - 1:
            self.current_x += 1
        elif self.next_action == 2 and self.current_y < self.height - 1:
            self.current_y += 1
        elif self.next_action == 3 and self.current_x > 0:
            self.current_x -=1


    def get_shortest_path(self):
        self.current_x = self.starting_x
        self.current_y = self.starting_y
        self.shortest_path = []
        self.shortest_path.append((self.current_x, self.current_y))
        while not self.is_terminal_state():
            self.get_next_action(1)
            self.next_locaction()
            self.shortest_path.append((self.current_x, self.current_y))
        return self.shortest_path


    def train(self, number_of_episodes):
        while number_of_episodes > 0:
            # self.get_starting_location() # losowo
            self.current_x = self.starting_x # od S
            self.current_y = self.starting_y # od S
            moves_of_episode = 0
            cumulative_reward_of_episode = 0
            while not self.is_terminal_state():
                moves_of_episode += 1
                self.get_next_action(self.epsilon)
                old_y, old_x = self.current_y, self.current_x
                self.next_locaction()
                cumulative_reward_of_episode += self.rewards[self.current_y, self.current_x]
                old_q_value = self.q_table[old_y, old_x, self.next_action]
                temporal_difference = self.rewards[self.current_y, self.current_x] + (self.discount_factor * np.max(self.q_table[self.current_y, self.current_x])) - old_q_value
                self.q_table[old_y, old_x, self.next_action] = old_q_value + (self.learning_rate * temporal_difference)
            self.training_moves.append(moves_of_episode)
            self.training_rewards.append(cumulative_reward_of_episode)
            number_of_episodes -= 1


def solve_maze(path, epsilon, discount_factor, learning_rate, train_episodes):
    maze = Q_learning(path)
    maze.from_file_to_array_maze()
    maze.epsilon = epsilon
    maze.discount_factor = discount_factor
    maze.learning_rate = learning_rate
    maze.maze_rewards()
    maze.train(train_episodes)
    return maze.get_shortest_path(), maze.training_moves, maze.training_rewards, train_episodes, path


def draw_and_print_resutls(solved_maze):
    plt.plot([n for n in range(solved_maze[3])], [m for m in solved_maze[1]])
    figure = plt.gcf()
    figure.savefig(f"{solved_maze[4][:-4]}moves.png", format="png")
    figure.clf()
    plt.plot([n for n in range(solved_maze[3])], [r for r in solved_maze[2]])
    figure.savefig(f"{solved_maze[4][:-4]}rewards.png", format="png")
    figure.clf()
    print(solved_maze[0])


solved_maze1 = solve_maze("example_maze1.txt", 0.9, 0.9, 0.9, 150)
solved_maze2 = solve_maze("example_maze2.txt", 0.9, 0.9, 0.9, 750)
solved_maze3 = solve_maze("example_maze3.txt", 0.9, 0.9, 0.9, 1500)
draw_and_print_resutls(solved_maze1)
draw_and_print_resutls(solved_maze2)
draw_and_print_resutls(solved_maze3)