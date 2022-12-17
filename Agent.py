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
        # self.production_data = []

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
        # print('self.qTable[1][self.currentState][(agent0Action, agent1Action)]',self.qTable[1][self.currentState][(agent0Action, agent1Action)])
        # self.timeStep += 1
        # print('the timestep is: ', self.timeStep)

    def chooseActionBasedOnQTable(self, currentState):
        # print('the currentStatein chooseActioncased on Qtable is: ',currentState)
        (m0, m1) = self.constructPayoffTable(currentState)
        probprob = lemkeHowson.lemkeHowson(m0, m1)
        # print("probprob: ",probprob)
        prob0 = np.array(probprob[0])
        re0 = np.where(prob0 == np.max(prob0))[0][0]
        prob1 = np.array(probprob[1])
        re1 = np.where(prob1 == np.max(prob1))[0][0]
        re = [re0, re1]
        # print('the re in chooseactionbasedonQtable is: ',re)
        actions_valid= self.get_validActions()
        # print('the actions_valid are: ', actions_valid)
        actionsAvailable = actions_valid[currentState]
        # for i in actionsAvailable:
        #     aAvailable= tuple(i)
        # print('the actionsavailable from chooseactionbasedonQtable are:',actionsAvailable)
        # print('actionsAvailable[re[self.agentIndex]]', actionsAvailable[re[self.agentIndex]][self.agentIndex])
        # c= None
        # for c in re[self.agentIndex]:
        #     break
        return actionsAvailable[re[self.agentIndex]][self.agentIndex]

    def step(self, agent0Action, agent1Action):
        part0, machine0 = agent0Action
        part1, machine1 = agent1Action
        # global nextState, reward
        # print('the step action is: ', action)
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
            self.mtp.b= [[0, 3, 3], [3, 0, 3], [3, 3, 3]]
            self.mtp.TTR =[20, 30, 25, 25]
            self.mtp.TBF=[10, 12, 12, 15]
            self.mtp.prod_count = np.zeros([n, ptypes])
            self.mtp.waiting_time= np.zeros(n)
            # self.production_data.append(self.mtp.prod_count)
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
                    if self.mtp.mprogress[machine0] >= 0 and self.mtp.Tr[machine0] == 0 and self.mtp.processing[machine0] == False and self.mtp.n_wait[machine0] == 1 and self.mtp.mready[machine0] == False:
                        self.mtp.run_gantry(agent0Action)
                        self.mtp.run_machine(machine0)
                    if self.mtp.mprogress[machine1] >= 0 and self.mtp.Tr[machine1] == 0 and self.mtp.processing[machine1] == False and self.mtp.n_wait[machine1] == 1 and self.mtp.mready[machine1] == False:
                        self.mtp.run_gantry(agent1Action)
                        self.mtp.run_machine(machine1)
            else:
                if np.any(self.mtp.mprogress == 0) and np.any(self.mtp.Tr == 0):
                    if False in self.mtp.processing and 1 in self.mtp.n_wait and False in self.mtp.mready:
                        # if list(self.mtp.mprogress).index(0)== list(self.mtp.Tr).index(0):
                        machine_to_be_unloaded= self.mtp.machine_to_be_unloaded()
                        self.mtp.unload_machine(machine_to_be_unloaded)
                        agent0Action= self.mtp.select_action(machine_to_be_unloaded)
                        agent1Action= self.chooseActionRandomly()
                        part, machine = agent0Action
                        self.mtp.plus_buffer(agent0Action)
                        self.mtp.mp[machine] = [0,0,0]
                        self.mtp.run_gantry(agent0Action)
                        self.mtp.run_machine(machine)
            self.reward= self.mtp.get_reward()
            self.mtp.t += 1
            self.nextState= self.mtp.get_state()
            if self.currentState != ():
                self.nashQLearning(self.gamma, agent0Action, self.reward, self.currentState, self.nextState, agent1Action, self.reward)
                # self.nashQLearning(self.gamma, agent0Action, self.reward, self.currentState, self.nextState, agent1Action, self.reward)
            self.currentState = self.nextState
            # print(f'the updated mtp parameters are: mprogress {self.mtp.mprogress}, ms: {self.mtp.ms}, mp: {self.mtp.mp}, n_wait: {self.mtp.n_wait}, processing: {self.mtp.processing}, prod_count: {self.mtp.prod_count}, cum_parts: {self.mtp.cum_parts}')
        # print(f'nextstate is: {self.nextState}, reward: {self.reward}, terminated: {self.mtp.get_terminated()}, info: {self.mtp.get_info()}')
        #     self.production_data.append(self.mtp.prod_count)
        #     self.mtp.get_prod_count()
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
    episodes = 0

    while episodes < 5:
        print ('the episode in run is: ', episodes)
        print('Prod.Count: ', agent_0.mtp.prod_count, 'waiting time: ', agent_0.mtp.waiting_time, 'reward is: ',agent_0.reward )
        # print('production dat: ', agent_0.production_data)
        # print('prod_counts: ', agent_0.mtp.get_prod_count())
        while True:
            agent0Action = agent_0.chooseActionRandomly()
            # print('the agent0action in run is: ', agent0Action)
            agent1Action = agent_1.chooseActionRandomly()
            while agent0Action[1]==agent1Action[1]:
                agent1Action = agent_1.chooseActionRandomly()
            # print('the agent1action in run is: ', agent1Action)
            nextState, reward, terminated, info = agent_0.step(agent0Action, agent1Action)
            # agent_0.nashQLearning(gamma, agent0Action, reward, currentState, nextState, agent1Action, reward)
            # agent_1.nashQLearning(gamma, agent0Action, reward, currentState, nextState, agent1Action, reward)
            # print('the reward_0, reward_1, nextState, terminated for simulation in run is: ', simulation(agent0Action, agent1Action))
            # print('the output of agent0.nashQLearning in run is: ',agent_0.nashQLearning(gamma, agent0Action,reward_0, currentState, nextState, agent1Action, reward_1))
            # print('the output of agent1.nashQLearning in run is: ',agent_1.nashQLearning(gamma, agent0Action, reward_0, currentState, nextState, agent1Action, reward_1))
            if agent_0.mtp.terminated== True:

            # if terminated == 1: # one episode of the game is end
                episodes += 1
                agent_0.currentState = resetStatus()
                agent_0.nextState = agent_0.currentState
                agent_0.mtp.t = 0
                break
            agent_0.currentState = agent_0.nextState

            # multiproduct_1.t += 1


# def test (agent_0 = agent, agent_1 = agent):
#     agent_0 = agent_0
#     agent_1 = agent_1
#     startState = resetStatus()
#     terminated = (0,0)
#     runs = 0
#     agentActionList = []
#     currentState = startState
#     # terminated = 0
#     while any(terminated) == 0:
#         agent0Action = agent_0.chooseActionBasedOnQTable(currentState)
#         # print('the agent0_action selected based on q table: ', agent0Action)
#         agent1Action = agent_1.chooseActionBasedOnQTable(currentState)
#         # print('the agent1_action selected based on q table: ', agent1Action)
#         agentActionList.append([agent0Action, agent1Action])
#         reward0, reward1, nextState, terminated = simulation(agent0Action, agent1Action)
#         currentState = nextState
#         # print('the new current state in test under while is: ',currentState)
#     agentActionList.append(currentState)
#     # print('the agent actionlist in test is: ', agentActionList)
#     return agentActionList

# runs = 0
# agentActionListEveryRun = {}
# agent_0 = agent(0)
# agent_1 = agent(1)
# run(agent_0, agent_1)
# agentActionListEveryRun[runs] = test(agent_0, agent_1)
# print ('the agentActionListEveryRun is:',agentActionListEveryRun)

# def rungame (agent_0 = agent, agent_1 = agent):
#     agent_0 = agent_0
#     agent_1 = agent_1
#     # print('the agent_0 and agent_1 in rungame are: ', agent_0, agent_1)
#     run(agent_0, agent_1)
#     runGameResult = test(agent_0, agent_1)
#     # print('the runGameResult is: ', runGameResult)
#     return runGameResult

if __name__ == "__main__":
    run(agent_0, agent_1)
    # agent0= agent(0)
    # all_states= agent0.get_allStates()
    # state= all_states[0]
    # print('the state is: ', state)
    # action= agent0.chooseActionRandomly()
    # payofftable= agent0.constructPayoffTable(state)
    # # q= agent0.initialSelfQTable()
    # # print(q)
    # # # playGameOne(agent_0, agent_1)
    # pool = multiprocessing.Pool(processes=2)
    # agentActionList = []
    # for i in range(2):
    #     agentActionList.append(pool.apply_async(rungame, (agent_0, agent_1)))
    # pool.close()
    # pool.join()

    # print (agent_0.qTable[0][3, 6])
    # print (agent_0.qTable[1][3, 6])
    # print (agent_1.qTable[0][8, 5])
    # print (agent_1.qTable[1][8, 5])
    # for res in agentActionList:
    #     print ('the res.get results in this: ',res.get())
