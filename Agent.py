import multiprocessing
import random
import NASH
import numpy as np
# import ProductionLine
import lemkeHowson
import matrix
from NASH import ms, null_part, n, ptypes, t, T, mp
from ProductionLine import Multiproduct
from itertools import permutations
import itertools

class agent:
    def __init__(self, agentIndex):  # initially the agent is not assigned
        # self.goalState = ()
        self.qTable = {}
        self.timeNumber = {}
        self.alpha = {}
        self.currentState = ()
        self.nextState = () #(1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1)
        self.strategy = {}
        self.agentIndex = agentIndex # agent number
        # self.startLocationIndex = startLocationIndex
        # self.locationIndex = startLocationIndex
        self.currentAction = 0 #(0,0)
        self.currentReward = 0
        self.timeStep = 0
        self.actions_all = []
        self.valid_actions={}
        self.reward= 0
        self.mtp= Multiproduct(ptypes, n, NASH.b, NASH.B, NASH.Tp, NASH.ng, NASH.MTTR, NASH.MTBF, T, NASH.Tl,
                                   NASH.Tu)
        self.actions= self.mtp.get_alist()
        # self.states= Multiproduct.get_state()
        self.actual_states = self.get_allStates()
        self.gamma=0.9
        self.num_episodes=10
        self.production_data = [[] for i in range(self.num_episodes)]
        self.waiting_data= [[] for i in range(self.num_episodes)]
        self.reward_data= [[] for i in range(self.num_episodes)]

    def get_validActions(self):
        for i in self.actual_states:
            self.valid_actions[i]=[self.actions]
        # print("The valid actions are: ", self.valid_actions)
        return self.valid_actions

    def get_allStates(self):  # only machine status and part type with machine are used as the states here. Make change in the prod. code
        a,b,c,d = [0, 1], [0, 1], [0, 1], [0, 1]   # each machine state: on/off
        m_stateLists = list(itertools.product(a, b, c, d))
        m_stateList = list(map(list, m_stateLists))
        a1,b1,c1,d1= [[0,0,0],[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0,0,0],[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0,0,0],[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0,0,0],[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # part with machine
        machine_partList = list(itertools.product(a1, b1, c1, d1))
        # self.my_states = tuple(itertools.product(m_stateList, machine_partList))
        my_states = list(itertools.product(m_stateList, machine_partList))
        # print(my_states[0])
        my_allstates = []
        for i in my_states:
            a = tuple(itertools.chain.from_iterable(i))
            bstates = tuple(y for x in a for y in (x if isinstance(x, list) else (x,)))
            my_allstates.append(bstates)
        actual_states = tuple(my_allstates)
        return actual_states

    def initialSelfStrategy(self):
        for i in self.actual_states:
            self.strategy[i] = {}
            for j in self.actions:
                self.strategy[i][j] = 0

    def initialSelfQTable(self):
        # agent0 and agnet1
        # each agent keeps two tables: one for himself and for the opponent
        # in Qtable, agent also can observe his opponent's action
        self.qTable[0] = {}
        self.qTable[1] = {}
        for i in self.actual_states:
            self.qTable[0][i] = {}
            self.qTable[1][i] = {}
            for j_1 in self.actions:
                for j_2 in self.actions:
                    self.qTable[0][i][(j_1, j_2)] = 0
                    self.qTable[1][i][(j_1, j_2)] = 0
        # return self.qTable[0], self.qTable[1]

    def initialSelfAlpha(self):
        # account the visiting number of each combination of states and actions
        for i in self.actual_states:
            self.alpha[i] = {}
            self.timeNumber[i] = {}
            for j_1 in self.actions:
                for j_2 in self.actions:
                    self.alpha[i][(j_1, j_2)] = 0
                    self.timeNumber[i][(j_1, j_2)] = 0

    def chooseActionRandomly(self):   # needs to check and update  # I may need to include the condition of gantry availability
        # choose action randomly
        # self.locationIndex = currentState[self.agentIndex]
        self.currentAction = random.choice(self.actions)
        # print('The current action is: ', self.currentAction)
        return self.currentAction

    def constructPayoffTable(self, state):
        # construct Payoff Table for agent 0 and agent 1
        # actions0 and actions1 are list for invalid actions
        # the content of Payoff Table is the Q value
        # actions0 = self.actions[state[0]]
        # actions1 = self.actions[state[1]]
        # print("The constrctPayoffTable state is: ", type(state))
        m0 = matrix.Matrix(len(self.actions), len(self.actions))
        m1 = matrix.Matrix(len(self.actions), len(self.actions))
        # print("m0 and m1: ", type(m0), type(m1))
        for i in range(len(self.actions)):
            for j in range(len(self.actions)):
                # print("i and j: ", self.actions[i],self.actions[j])
                m0.setItem(i + 1, j + 1, self.qTable[0][state][(self.actions[i], self.actions[j])])
                m1.setItem(i + 1, j + 1, self.qTable[1][state][(self.actions[i], self.actions[j])])
        # print('payoff tables for m0: ',m0, 'payofftable for m1: ', m1 )
        return m0, m1

    def nashQLearning(self, gamma, agent0Action, agent0Reward, currentState, nextState, agent1Action, agent1Reward):
        # print(f'The agent0action in nashQlearning is {agent0Action}, agent0reward is {agent0Reward}, currentstate is {currentState},\
        #  nextstate is {nextState} agent1action is {agent1Action} and agent1reward {agent1Reward}')
        self.gamma = gamma  # gamma is actually alpha of the equation
        # self.currentState = currentState
        # self.nextState = nextState
        self.timeNumber[self.currentState][(agent0Action, agent1Action)] += 1
        self.alpha[self.currentState][(agent0Action, agent1Action)] = 1.0 / self.timeNumber[self.currentState][(agent0Action, agent1Action)]
        # m0 and m1 are payoff tables of agent0 and agent1
        (m0, m1) = self.constructPayoffTable(self.nextState)
        probprob = lemkeHowson.lemkeHowson(m0, m1)
        # print('The probprob is: ', probprob)
        prob0 = np.array(probprob[0])
        # print('The prob0 is: ', prob0)
        prob1 = np.array(probprob[1])
        # print('The prob1 is: ', prob1)
        prob0 = np.matrix(prob0)
        prob1 = np.matrix(prob1).reshape((-1, 1))
        # calculate the nash values
        m_m0 = []
        m_m1 = []
        for i in range(m0.getNumRows()):
            for j in range(m0.getNumCols()):
                m_m0.append(m0.getItem(i + 1, j + 1))
        for i in range(m1.getNumRows()):
            for j in range(m1.getNumCols()):
                m_m1.append(m1.getItem(i + 1, j + 1))
        m_m0 = np.matrix(m_m0).reshape((m0.getNumRows(), m0.getNumCols()))
        m_m1 = np.matrix(m_m1).reshape((m1.getNumRows(), m1.getNumCols()))
        m_nash0 = prob0 * m_m0 * prob1
        m_nash1 = prob0 * m_m1 * prob1
        nash0 = m_nash0[0, 0].nom() / m_nash0[0, 0].denom()
        nash1 = m_nash1[0, 0].nom() / m_nash1[0, 0].denom()
        nashQValues = [nash0, nash1]
        # print('the nashQvalues are: ', nashQValues)
        action_tuple= (agent0Action, agent1Action)
        alpha_var = self.alpha[self.currentState][action_tuple]
        self.qTable[0][self.currentState][action_tuple]= (1 - alpha_var)* self.qTable[0][self.currentState][action_tuple] + alpha_var * (agent0Reward + self.gamma * nashQValues[0])
        # print('self.qTable[0][self.currentState][(agent0Action, agent1Action)]', self.qTable[0][self.currentState][(agent0Action, agent1Action)])
        self.qTable[1][self.currentState][action_tuple] = (1 - alpha_var) * self.qTable[1][self.currentState][action_tuple]+ alpha_var * (agent1Reward + self.gamma * nashQValues[1])

    def chooseActionBasedOnQTable(self, currentState):
        # print('the currentStatein chooseActioncased on Qtable is: ',currentState)
        (m0, m1) = self.constructPayoffTable(currentState)
        probprob = lemkeHowson.lemkeHowson(m0, m1)
        prob0 = np.array(probprob[0])
        re0 = np.where(prob0 == np.max(prob0))[0][0]
        prob1 = np.array(probprob[1])
        re1 = np.where(prob1 == np.max(prob1))[0][0]
        re = [re0, re1]
        actions_valid= self.get_validActions()
        actionsAvailable = actions_valid[currentState]
        return actionsAvailable[re[self.agentIndex]][self.agentIndex]

    def step(self, agent0Action, agent1Action, episode):
        part0, machine0 = agent0Action
        part1, machine1 = agent1Action
        self.currentState= self.nextState  #self.mtp.get_state()  # check this out if we need to get the state here, should it be equal to the self.nextstate
        if self.mtp.t == 0:
            self.mtp.n_SB= [0, 0, 0, 0]
            self.mtp.mp= [[0,0,0],[0,0,0],[0,0,0],[0,0,0]] #np.zeros([n, ptypes])
            self.mtp.processing=[False, False, False, False]
            self.mtp.n_wait = [1, 1, 1, 1]
            self.mtp.gs= [0, 0, 0, 0]
            self.mtp.ms= [1, 1, 1, 1]
            self.mtp.mprogress= np.zeros(n)
            self.mtp.mready= [False, False, False, False]
            self.mtp.loading= np.zeros(n)
            self.mtp.unloading = np.zeros(n)
            self.mtp.g_load = np.zeros(n)
            self.mtp.g_unload = np.zeros(n)
            self.mtp.Tr = np.zeros(n)
            self.mtp.b= [[0, 5, 5], [5, 0, 5], [5, 5, 5]]
            self.mtp.TTR =[20, 30, 25, 25]
            self.mtp.TBF=[100, 120, 125, 150]
            self.mtp.prod_count = np.zeros([n, ptypes])
            self.mtp.waiting_time= np.zeros(n)
            self.mtp.downtime = np.zeros(n)
            self.mtp.W = np.zeros(n)
            self.production_data[episode].append(self.mtp.prod_count.copy())
            self.waiting_data[episode].append(self.mtp.waiting_time.copy())
            self.reward_data[episode].append(self.reward)
        while self.mtp.t < T:
            # self.production_data.append(self.mtp.prod_count)
            [self.mtp.run_machine(k) for k in range(n)]
            if self.mtp.t!=0 and self.currentState != () and np.sum(self.mtp.gs)<2 and self.mtp.mp_has_nullPart()==True:
                # agent0Action = self.chooseActionRandomly()
                # agent1Action = self.chooseActionRandomly()
                agent0Action = self.chooseActionBasedOnQTable(self.currentState)
                agent1Action = self.chooseActionBasedOnQTable(self.currentState)
                while agent0Action[1] == agent1Action[1]:
                    agent0Action = self.chooseActionRandomly()
                    agent1Action = self.chooseActionRandomly()
                # action= self.chooseActionBasedOnQTable(self.currentState)
                # part, machine = action
                part0, machine0 = agent0Action
                part1, machine1 = agent1Action
            # if np.any(mp== null_part):
            if self.mtp.mp_has_nullPart()==True:
                # part, machine = action
                mp_1= np.array(self.mtp.mp)
                null_part_1= np.array([0,0,0])
                if np.all(mp_1[machine0]== null_part_1) or np.all(mp_1[machine1]== null_part_1):
                    self.mtp.get_W()
                    if self.mtp.W[machine0]==0:
                        self.mtp.run_gantry(agent0Action)
                        self.mtp.run_machine(machine0)
                    else:
                        self.mtp.downtime[machine0]+=1
                        self.mtp.total_downtime[machine0]+=1
                    if self.mtp.W[machine1]==0:
                        self.mtp.run_gantry(agent1Action)
                        self.mtp.run_machine(machine1)
                    else:
                        self.mtp.downtime[machine1]+=1
                        self.mtp.total_downtime[machine1]+=1
                else:
                    self.mtp.get_W()
                    if self.mtp.W[machine0] == 0:
                        if self.mtp.mprogress[machine0] >= 0 and self.mtp.Tr[machine0] == 0 and self.mtp.processing[machine0] == False and self.mtp.n_wait[machine0] == 1 and self.mtp.mready[machine0] == False:
                            self.mtp.run_gantry(agent0Action)
                            self.mtp.run_machine(machine0)
                    else:
                        self.mtp.downtime[machine0] += 1
                        self.mtp.total_downtime[machine0] += 1
                    if self.mtp.W[machine1] == 0:
                        if self.mtp.mprogress[machine1] >= 0 and self.mtp.Tr[machine1] == 0 and self.mtp.processing[machine1] == False and self.mtp.n_wait[machine1] == 1 and self.mtp.mready[machine1] == False:
                            self.mtp.run_gantry(agent1Action)
                            self.mtp.run_machine(machine1)
                    else:
                        self.mtp.downtime[machine1] += 1
                        self.mtp.total_downtime[machine1] += 1
            else:
                self.mtp.get_W()
                if np.any(self.mtp.mprogress == 0) and np.any(self.mtp.Tr == 0):
                    if False in self.mtp.processing and 1 in self.mtp.n_wait and False in self.mtp.mready:
                        # if list(self.mtp.mprogress).index(0)== list(self.mtp.Tr).index(0):
                        machine_to_be_unloaded= self.mtp.machine_to_be_unloaded()
                        if (np.sum(self.mtp.b[machine_to_be_unloaded-1])>0 or machine_to_be_unloaded==0) and self.mtp.next_buffer_has_space(machine_to_be_unloaded)==True:
                            self.mtp.unload_machine(machine_to_be_unloaded)
                            agent0Action= self.mtp.select_action(machine_to_be_unloaded)
                            agent1Action= self.chooseActionRandomly()
                            part, machine = agent0Action
                            self.mtp.plus_buffer(agent0Action)
                            self.mtp.mp[machine] = [0,0,0]
                            self.mtp.run_gantry(agent0Action)
                            self.mtp.run_machine(machine)
                        else:
                            machine_to_be_unloaded = self.mtp.next_machine_to_be_unloaded()
                            if machine_to_be_unloaded != None:
                                if (np.sum(self.mtp.b[machine_to_be_unloaded - 1]) > 0 or machine_to_be_unloaded == 0) and self.mtp.next_buffer_has_space(machine_to_be_unloaded) == True:
                                    self.mtp.unload_machine(machine_to_be_unloaded)
                                    agent0Action = self.mtp.select_action(machine_to_be_unloaded)
                                    agent1Action = self.chooseActionRandomly()
                                    part, machine = agent0Action
                                    self.mtp.plus_buffer(agent0Action)
                                    self.mtp.mp[machine] = [0, 0, 0]
                                    self.mtp.run_gantry(agent0Action)
                                    self.mtp.run_machine(machine)

            self.reward= self.mtp.get_reward()
            self.mtp.t += 1
            self.nextState= self.mtp.get_state()
            if self.currentState != ():
                self.nashQLearning(self.gamma, agent0Action, self.reward, self.currentState, self.nextState, agent1Action, self.reward)
                # self.nashQLearning(self.gamma, agent0Action, self.reward, self.currentState, self.nextState, agent1Action, self.reward)
            self.currentState = self.nextState
            self.production_data[episode].append(self.mtp.prod_count.copy())
            self.waiting_data[episode].append(self.mtp.waiting_time.copy())
            self.reward_data[episode].append(self.reward)
        return self.nextState, self.reward, self.mtp.get_terminated(), self.mtp.get_info()

multiproduct_1= Multiproduct(ptypes, n, NASH.b, NASH.B, NASH.Tp, NASH.ng, NASH.MTTR, NASH.MTBF, T, NASH.Tl,NASH.Tu)
agent_0 = agent(0)
agent_1 = agent(1)

def resetStatus():
    # ms = [1,1,1,1]
    # mp= np.zeros([n,ptypes])
    resetState= (1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0)
    return resetState

def run(agent_0 = agent, agent_1 = agent):
    gamma = 0.9
    agent_0 = agent_0
    agent_1 = agent_1
    currentState = multiproduct_1.get_state()
    timeStep = 0 # calculate the timesteps in one episode
    terminated = 0
    agent_0.initialSelfQTable()
    agent_1.initialSelfQTable()
    agent_0.initialSelfAlpha()
    agent_1.initialSelfAlpha()
    episode = 0

    while episode < 20:    #agent_0.num_episodes:
        print ('the episode in run is: ', episode)
        print('Prod.Count: ', agent_0.mtp.prod_count, 'waiting time: ', agent_0.mtp.waiting_time, 'reward is: ',agent_0.reward )
        # print('production data: ', agent_0.production_data)
        prod_data = np.array(agent_0.production_data, dtype= object)
        # max_prod = prod_data.max()
        print('prod_data shape: ',prod_data.shape)
        # print('prod_counts: ', agent_0.mtp.get_prod_count())
        while True:
            agent0Action = agent_0.chooseActionRandomly()
            # print('the agent0action in run is: ', agent0Action)
            agent1Action = agent_1.chooseActionRandomly()
            while agent0Action[1]==agent1Action[1]:
                agent1Action = agent_1.chooseActionRandomly()
            # print('the agent1action in run is: ', agent1Action)
            nextState, reward, terminated, info = agent_0.step(agent0Action, agent1Action, episode)
            if agent_0.mtp.terminated== True:
                episode += 1
                agent_0.currentState = resetStatus()
                agent_0.nextState = agent_0.currentState
                agent_0.mtp.t = 0
                break
            agent_0.currentState = agent_0.nextState

    import matplotlib.pyplot as plt
    import seaborn as sns
    np.set_printoptions(threshold= np.inf)

    def plot_std_shade(curves_list,title,xlabel,ylabel, c='b'):

        curves=np.array(curves_list)
        mean_curve=curves.mean(axis=0)
        std_curve=curves.std(axis=0)
        min_curve= curves.min(axis=0)
        max_curve= curves.max(axis=0)


        N=curves.shape[1]
        x=np.arange(N)
        plt.plot(x,mean_curve,c+'-',label=title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.fill_between(x,min_curve,max_curve,color=c,alpha=0.3)
        # plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.2)
        plt.legend()
        plt.ylabel(ylabel)
        # plt.fill_between(x,mean_curve-std_curve,mean_curve+std_curve,color=c,alpha=0.2)
        # plt.legend()

    sns.set()
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
    prod_data= np.array(agent_0.production_data)
    max_prod = prod_data.max()
    print('production data shape: ', prod_data.shape)
    # print('production_data',agent_0.production_data[0])
    plt.figure(figsize=(20,4))
    for e in range(prod_data.shape[0]):
        for p in range(prod_data.shape[3]):
            plt.subplot(1,prod_data.shape[0], e+1)
            plt.plot(prod_data[e][:,-1,p], label= f"part= {p}", color= colors[p])
        plt.title(f"episode={e}")
        plt.ylim(0,max_prod)
        plt.legend()
    plt.show()
    #
    for p in range(prod_data.shape[3]):
        curve_list= prod_data[:,:,-1,p]
        plot_std_shade(curve_list, f"part= {p}", 'time_steps', 'prod_count', colors[p])
    plt.show()

    waiting_data= np.array(agent_0.waiting_data)
    max_wait = waiting_data.max()
    print('waiting_data shape: ',waiting_data.shape)
    # print('waiting_data', agent_0.waiting_data[0])
    plt.figure(figsize=(20, 4))
    for e in range(waiting_data.shape[0]):
        for m in range(waiting_data.shape[2]):
            plt.subplot(1, waiting_data.shape[0], e + 1)
            plt.plot(waiting_data[e][:, m], label=f"machine= {m}", color=colors[m])
        plt.title(f"episode={e}")
        plt.ylim(0, max_wait)
        plt.legend()
    plt.show()
    #
    for m in range(waiting_data.shape[2]):
        curve_list = waiting_data[:, :, m]
        plot_std_shade(curve_list, f"machine= {m}", 'time_steps', 'waiting_time', colors[m])
    plt.show()


    reward_data = np.array(agent_0.reward_data)
    max_reward = reward_data.max()
    min_reward = reward_data.min()
    print('reward_data_shape: ',reward_data.shape)
    # print('reward_data', agent_0.reward_data[0])
    plt.figure(figsize=(20, 4))
    for e in range(reward_data.shape[0]):
        plt.subplot(1, reward_data.shape[0], e + 1)
        plt.plot(reward_data[e][1:])
        plt.title(f"episode={e}")
        plt.ylim(min_reward, max_reward)
        plt.legend()
    plt.show()
    #
    x = np.arange(1,reward_data.shape[0]+1, dtype= int)
    plt.plot(x,reward_data[:, -1])
    plt.show()

# production files
    with open('prod_count_0.csv', 'w+') as prod_file:
        result_output = ''
        result_output += str(prod_data[0])
        prod_file.write(result_output)
    with open('prod_count_1.csv', 'w+') as prod_file:
        result_output = ''
        result_output += str(prod_data[1])
        prod_file.write(result_output)
    with open('prod_count_2.csv', 'w+') as prod_file:
        result_output = ''
        result_output += str(prod_data[2])
        prod_file.write(result_output)
    with open('prod_count_3.csv', 'w+') as prod_file:
        result_output = ''
        result_output += str(prod_data[3])
        prod_file.write(result_output)

# waiting files
    with open('waiting_data_0.csv', 'w+') as waiting_file:
        result_output = ''
        result_output += str(waiting_data[0])
        waiting_file.write(result_output)
    with open('waiting_data_1.csv', 'w+') as waiting_file:
        result_output = ''
        result_output += str(waiting_data[1])
        waiting_file.write(result_output)
    with open('waiting_data_2.csv', 'w+') as waiting_file:
        result_output = ''
        result_output += str(waiting_data[2])
        waiting_file.write(result_output)
    with open('waiting_data.csv_3', 'w+') as waiting_file:
        result_output = ''
        result_output += str(waiting_data[3])
        waiting_file.write(result_output)

# reward files
    with open('reward_data.csv', 'w+') as reward_file:
        result_output = ''
        result_output += str(reward_data)
        reward_file.write(result_output)


if __name__ == "__main__":
    run(agent_0, agent_1)


