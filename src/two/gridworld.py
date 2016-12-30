import numpy as np

class GridWorld:

    def __init__(self, tot_row, tot_col):
        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col
        #The world is a matrix of size row x col x 2
        #The first layer contains the obstacles
        #The second layer contains the rewards
        #self.world_matrix = np.zeros((tot_row, tot_col, 2))
        self.transition_matrix = np.ones((self.action_space_size, self.action_space_size))/ self.action_space_size
        #self.transition_array = np.ones(self.action_space_size) / self.action_space_size
        self.reward_matrix = np.zeros((tot_row, tot_col))
        self.state_matrix = np.zeros((tot_row, tot_col))
        self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]

    def setTransitionArray(self, transition_array):
        if(transition_array.shape != self.transition_array):
            raise ValueError('The shape of the two matrices must be the same.') 
        self.transition_array = transition_array        

    def setTransitionMatrix(self, transition_matrix):
        if(transition_matrix.shape != self.transition_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.') 
        self.transition_matrix = transition_matrix

    def setRewardMatrix(self, reward_matrix):
        '''Set the reward matrix.

        '''
        self.reward_matrix = reward_matrix

    def setStateMatrix(self, state_matrix):
        '''Set the obstacles in the world.

        The input to the function is a matrix with the
        same size of the world 
        -1 for states which are not walkable.
        +1 for terminal states
         0 for all the walkable states (non terminal)
        '''
        self.state_matrix = state_matrix

    def setPosition(self, index_row=None, index_col=None):
        if(index_row is None or index_col is None): self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]
        else: self.position = [index_row, index_col]

    def render(self):
        graph = ""
        for row in range(self.world_row):
            row_string = ""
            for col in range(self.world_col):
                if(self.state_matrix[row, col] == 0): row_string += ' - '
                elif(self.state_matrix[row, col] == -1): row_string += ' # '
                elif(self.state_matrix[row, col] == +1): row_string += ' * '
            row_string += '\n'
            graph += row_string 
        print graph            

    def reset(self):
        ''' Set the position of the robot in the bottom left corner.

        '''
        self.position = [self.world_row-1, 0]

    def step(self, action):
        ''' One step in the world.

        observation, reward, done = env.step(action)
        The robot moves one step in the world based on the action given.
        The action can be 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        @return observation the position of the robot after the step
        @return reward the reward associated with the next state
        @return done it is True if the state is terminal  
        '''
        if(action >= self.action_space_size): 
            raise ValueError('The action is not included in the action space.')

        #Based on the current action and the probability derived
        #from the trasition model it chooses a new actio to perform
        action = np.random.choice(4, 1, p=self.transition_matrix[action])
        #action = self.transition_model(action)

        #Generating a new position based on the current position and action
        if(action == 0): new_position = [self.position[0]+1, self.position[1]]
        elif(action == 1): new_position = [self.position[0], self.position[1]+1]
        elif(action == 2): new_position = [self.position[0]-1, self.position[1]]
        elif(action == 3): new_position = [self.position[0], self.position[1]-1]

        #Check if the new position is a valid position
        print(self.state_matrix)
        if (new_position[0]>=0 and new_position[0]<self.world_row):
            if(new_position[1]>=0 and new_position[1]<self.world_col):
                if(self.state_matrix[new_position[0], new_position[1]] != -1):
                    self.position = new_position

        reward = self.reward_matrix[self.position[0], self.position[1]]
        #Done is True if the state is a terminal state
        done = bool(self.state_matrix[self.position[0], self.position[1]])
        return self.position, reward, done









