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

    def get_validActions(self):
        for i in self.actual_states:
            self.valid_actions[i]=[self.actions]
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
        return self.currentAction

    def constructPayoffTable(self, state):
        # construct Payoff Table for agent 0 and agent 1
        # actions0 and actions1 are list for invalid actions
        # the content of Payoff Table is the Q value
        # actions0 = self.actions[state[0]]
        # actions1 = self.actions[state[1]]
        # print("payoff state: ", state)
        m0 = matrix.Matrix(len(self.actions), len(self.actions))
        m1 = matrix.Matrix(len(self.actions), len(self.actions))
        # print("m0 and m1: ", m0, m1)
        for i in range(len(self.actions)):
            for j in range(len(self.actions)):
                # print("i and j: ", self.actions[i],self.actions[j])
                m0.setItem(i + 1, j + 1, self.qTable[0][state][(self.actions[i], self.actions[j])])
                m1.setItem(i + 1, j + 1, self.qTable[1][state][(self.actions[i], self.actions[j])])
        return m0, m1

    def nashQLearning(self, gamma, agent0Action, agent0Reward, currentState, nextState, agent1Action, agent1Reward):
        self.gamma = gamma  # gamma is actually alpha of the equation
        self.currentState = currentState
        self.nextState = nextState
        self.timeNumber[self.currentState][(agent0Action, agent1Action)] += 1
        self.alpha[self.currentState][(agent0Action, agent1Action)] = 1.0 / self.timeNumber[self.currentState][(agent0Action, agent1Action)]
        # m0 and m1 are payoff tables of agent0 and agent1
        (m0, m1) = self.constructPayoffTable(nextState)
        probprob = lemkeHowson.lemkeHowson(m0, m1)
        prob0 = np.array(probprob[0])
        prob1 = np.array(probprob[1])
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
        self.qTable[0][self.currentState][(agent0Action, agent1Action)] \
            = (1 - self.alpha[self.currentState][(agent0Action, agent1Action)]) \
              * self.qTable[0][self.currentState][(agent0Action, agent1Action)] \
              + self.alpha[self.currentState][(agent0Action, agent1Action)] \
              * (agent0Reward + self.gamma * nashQValues[0])
        print(self.qTable[0][self.currentState][(agent0Action, agent1Action)])
        self.qTable[1][self.currentState][(agent0Action, agent1Action)] \
            = (1 - self.alpha[self.currentState][(agent0Action, agent1Action)]) \
              * self.qTable[1][self.currentState][(agent0Action, agent1Action)] \
              + self.alpha[self.currentState][(agent0Action, agent1Action)] \
              * (agent1Reward + self.gamma * nashQValues[1])
        print(self.qTable[1][self.currentState][(agent0Action, agent1Action)])
        self.timeStep += 1

    def chooseActionBasedOnQTable(self, currentState):
        print('currentState',currentState)
        (m0, m1) = self.constructPayoffTable(currentState)
        probprob = lemkeHowson.lemkeHowson(m0, m1)
        print("probprob: ",probprob)
        prob0 = np.array(probprob[0])
        re0 = np.where(prob0 == np.max(prob0))[0][0]
        prob1 = np.array(probprob[1])
        re1 = np.where(prob1 == np.max(prob1))[0][0]
        re = [re0, re1]
        print(re)
        actions_valid= self.get_validActions()
        actionsAvailable = actions_valid[currentState[self.agentIndex]]
        print(actionsAvailable)
        return actionsAvailable[re[self.agentIndex]]

    def step(self, action):
        # part, machine = action
        # global nextState, reward
        # print('the step action is: ', action)
        self.currentState= self.mtp.get_state()
        if self.mtp.t == 0:
            self.mtp.n_SB= [0, 0, 0, 0]
            self.mtp.mp= np.zeros([n, ptypes])
            self.mtp.processing=[False, False, False, False]
            self.mtp.n_wait = [True, True, True, True]
            self.mtp.gs= [0, 0, 0, 0]
            self.mtp.ms= [1, 1, 1, 1]

        while self.mtp.t < T:
            [self.mtp.run_machine(k) for k in range(n)]
            if np.any(mp== null_part):
                part, machine = action
                if all(mp[machine]== null_part):
                    self.mtp.get_W()
                    if self.mtp.W[machine]==0:
                        self.mtp.run_gantry(action)
                        self.mtp.run_machine(machine)
                    else:
                        self.mtp.downtime[machine]+=1
                        self.mtp.total_downtime[machine]+=1
                else:
                    if self.mtp.mprogress[machine] >= 0 and self.mtp.Tr[machine] == 0 and self.mtp.processing[machine] == False and self.mtp.n_wait[machine] == 1 and self.mtp.mready[machine] == False:
                        self.mtp.run_gantry(action)
                        self.mtp.run_machine(machine)
            else:
                if any(self.mtp.mprogress) == 0 and any(self.mtp.Tr) == 0 and any(self.mtp.processing) == False and \
                        any(self.mtp.n_wait) == 1 and any(self.mtp.mready) == False:
                    action = agent(self.agentIndex).chooseActionBasedOnQTable(self.currentState)
                    part, machine = action
                    self.mtp.run_gantry(action)
                    self.mtp.run_machine(machine)
            self.reward= self.mtp.get_reward()
            self.mtp.t += 1
            self.nextState= self.mtp.get_state()
        return self.nextState, self.reward, self.mtp.get_terminated(), self.mtp.get_info()

multiproduct_1= Multiproduct(ptypes, n, NASH.b, NASH.B, NASH.Tp, NASH.ng, NASH.MTTR, NASH.MTBF, T, NASH.Tl,NASH.Tu)
# reward = multiproduct_1.get_reward()
# nextState=()
# agent_0= agent(0)
def simulation(action_0, action_1):
    # action_0 = action_0
    # action_1 = action_1
    # currentState = multiproduct_1.get_state()
    # nextState= agent_1.nextState

    # terminated = 0 #endgameflag
    nextState, reward_0, terminated_0, info_0= agent_0.step(action_0)
    nextState, reward_1, terminated_1, info_1= agent_1.step(action_1)

    terminated= (terminated_0, terminated_1)
    return reward_0, reward_1, nextState, terminated


def resetStatus():
    # ms = [1,1,1,1]
    # mp= np.zeros([n,ptypes])
    resetState= (1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0)
    return resetState


agent_0 = agent(0)
agent_1 = agent(1)

def run(agent_0 = agent, agent_1 = agent):
    gamma = 0.9
    agent_0 = agent_0
    agent_1 = agent_1
    currentState = multiproduct_1.get_state()
    timeStep = 0 # calculate the timesteps in one episode
    episodes = 0
    terminated = False
    agent_0.initialSelfQTable()
    agent_1.initialSelfQTable()
    agent_0.initialSelfAlpha()
    agent_1.initialSelfAlpha()
    while episodes < 2: #100
        print (episodes)
        while True:
            agent0Action = agent_0.chooseActionRandomly()
            agent1Action = agent_1.chooseActionRandomly()
            reward_0, reward_1, nextState, terminated = simulation(agent0Action, agent1Action)
            agent_0.nashQLearning(gamma, agent0Action,reward_0, currentState, nextState, agent1Action, reward_1)
            agent_1.nashQLearning(gamma, agent0Action, reward_0, currentState, nextState, agent1Action, reward_1)
            if (terminated == True): # one episode of the game is end
                episodes += 1
                currentState = resetStatus()
                break
            currentState = nextState


def test (agent_0 = agent, agent_1 = agent):
    agent_0 = agent_0
    agent_1 = agent_1
    startState = resetStatus()
    terminated = 0
    runs = 0
    agentActionList = []
    currentState = startState
    terminated = 0
    while terminated != 1:
        agent0Action = agent_0.chooseActionBasedOnQTable(currentState)
        agent1Action = agent_1.chooseActionBasedOnQTable(currentState)
        agentActionList.append([agent0Action, agent1Action])
        reward0, reward1, nextState, terminated = simulation(agent0Action, agent1Action)
        currentState = nextState
    agentActionList.append(currentState)
    return agentActionList

runs = 0
agentActionListEveryRun = {}
# agent_0 = agent(0)
# agent_1 = agent(1)
run(agent_0, agent_1)
agentActionListEveryRun[runs] = test(agent_0, agent_1)
print (agentActionListEveryRun)

def rungame (agent_0 = agent, agent_1 = agent):
    agent_0 = agent_0
    agent_1 = agent_1
    run(agent_0, agent_1)
    runGameResult = test(agent_0, agent_1)
    return runGameResult

if __name__ == "__main__":
    # agent0= agent(0)
    # all_states= agent0.get_allStates()
    # state= all_states[0]
    # print('the state is: ', state)
    # action= agent0.chooseActionRandomly()
    # payofftable= agent0.constructPayoffTable(state)
    # # q= agent0.initialSelfQTable()
    # # print(q)
    # # # playGameOne(agent_0, agent_1)
    pool = multiprocessing.Pool(processes=2)
    agentActionList = []
    for i in range(2):
        agentActionList.append(pool.apply_async(rungame, (agent_0, agent_1)))
    pool.close()
    pool.join()

    # print (agent_0.qTable[0][3, 6])
    # print (agent_0.qTable[1][3, 6])
    # print (agent_1.qTable[0][8, 5])
    # print (agent_1.qTable[1][8, 5])
    for res in agentActionList:
        print (res.get())
