import random
import copy
import numpy as np
from NASH import T, n, ptypes, p_sequence, parts, mp, n_wait, n_SB, D, s_criticalIndx, B, b, Tp, MTTR, ng, MTBF, gs, ms, \
    Tl, Tu, t, null_part, omega_d, omega_s


class Multiproduct:

    def __init__(self, ptypes, n, b, B, Tp, ng, MTTR, MTBF, T, Tl, Tu):
        # self.m = m
        self.ptypes = ptypes  # product_types
        self.n = n  # n_machines
        self.b = b  # buffer_level
        self.B = B  # buffer_capacity
        self.Tp = Tp  # process_time
        self.Tl = Tl  # load_time
        self.Tu = Tu  # unload_time
        self.ng = ng  # n_gantries
        self.MTTR = MTTR
        self.MTBF = MTBF
        self.ms = ms  # machine_state
        self.gs = gs  # gantry_state
        self.T = T  # simulation time
        self.t=0
        # self.machines = machines
        self.n_SB = n_SB
        self.n_wait = n_wait
        self.processing = [False, False, False, False]  # machine busy or not
        self.p_sequence = p_sequence  # sequence of each part through the line
        self.mp = mp
        self.parts = parts
        self.prod_count = np.zeros([n, ptypes])
        self.cum_parts= np.zeros(n)  # cumulative parts produced by each machine
        self.TBF = copy.deepcopy(MTBF)
        self.TTR = copy.deepcopy(MTTR)
        self.status=True  # just for code checkup, it s also machine status
        self.W= np.zeros(n) # machine downtime, code checkup
        self.terminated = False
        self.Tr= np.zeros(n)
        self.waiting_time = np.zeros(n)
        self.prod_count = np.zeros([n, ptypes])
        self.mready=[False, False, False, False]   # if ready to process or not
        self.mprogress = np.zeros(n)  # processed parts on each machine irrespective of type
        self.downtime= np.zeros(n)
        self.total_downtime =  np.zeros(n)
        self.g_load= np.zeros(n) # which machine is being loaded
        self.g_unload= np.zeros(n)  # which machine is being unloaded
        self.loading= np.zeros(n)
        self.unloading = np.zeros(n)
        self.unload_check= [False, False, False, False]

    def get_alist(self):  # action list
        self.actionList = []
        for i in range(ptypes):
            for j in range(n):
                if p_sequence[i][j] == 1:
                    self.actionList.append((i, j))
        # action_selected= random.choice(self.actionList)
        return self.actionList

    def run_machine(self, j):  # j is the machine number

        if np.any(self.mp[j] > np.zeros(ptypes)): # if machine has a part
            if self.ms[j] == 1: # if machine is running or down
                self.downtime[j] = 0
                if self.mready[j]==True:  # processing not started yet, part loaded,
                    if self.Tr[j] == 0:
                        self.processing[j] = True  # start part processing, initiating the part processing
                        self.mprogress[j]+= 1/Tp[j]
                        self.Tr[j] += 1
                    else:
                        self.mprogress[j] += 1 / Tp[j]
                        self.Tr[j] += 1
                        if self.mprogress[j]>=1 or self.Tr[j]==self.Tp[j]:
                            self.mprogress[j]=0
                            self.Tr[j]=0
                            self.cum_parts[j]+=1
                            self.processing[j]= False
                            self.n_wait[j]=1
                            self.mready[j]=False
                            self.waiting_time[j]+= 1
                else:
                    if self.n_wait[j]==0 and self.processing[j]==True:
                        self.loading[j] += 1
                        self.g_load[j]=1
                        if self.loading[j] == self.Tl[j]:
                            self.loading[j] = 0
                            self.gs[j] = 0
                            self.g_load[j] = 0
                            self.mready[j]= True
                    # if self.n_wait[j]==0 and self.processing[j]==True and self.unload_check[j]==True:
                    #     self.g_unload[j] = 1
                    #     self.unloading[j] += 1
                    #     if self.unloading[j] == self.Tu[j]:
                    #         self.unloading[j] = 0
                    #         self.g_unload[j] = 0
                    #         self.unload_check[j]= False
            else:
                self.downtime[j]+=1
                self.total_downtime += 1
        else:
            self.n_wait[j] = 1  # machine j is waiting to be loaded
            self.processing[j] = False
            self.waiting_time[j]+=1
            self.mready[j]= False

        return self.Tr[j], self.cum_parts[j], self.waiting_time[j], self.downtime[j], self.n_wait[j]

    def load_machine(self,j):
        if self.n_wait[j]==1:
            if np.all(self.mp[j]==np.zeros(ptypes)):
                if j!=0:
                    if np.any(self.b[j - 1]) > 0:
                        self.gs[j]=1
                        self.g_load[j]=1
                    else:
                        self.n_SB[j]=1
                        print('previous buffer is empty')
                else:
                    self.gs[j] = 1
                    self.g_load[j] = 1
            else:
                print('machine has already a part')
        else:
            print('machine needs to be in waiting')

    def unload_machine(self, j):
        if self.n_wait[j] == 1:
            if np.any(self.mp[j] > np.zeros(ptypes)):
                if j!=(n-1):
                    if np.sum(self.b[j])< self.B[j]:
                        self.gs[j]=1
                    else:
                        self.n_SB[j]=1
                        print('next buffer is full')
                else:
                    self.gs[j] = 1
            else:
                print('machine is already empty')
        else:
            print('machine needs to be in waiting')

    def run_gantry(self, action):
        part, machine = action
        if (np.sum(gs)<ng) and self.gs[machine] == 0:  # if gantry is available and machine is not assigned a gantry
            if self.ms[machine]==1:  # if machine to be loaded is up
                if np.sum(self.mp[machine])==0:    # machine has no part and is waiting to be loaded
                    if not self.processing[machine] and not self.mready[machine] and self.n_SB[machine]==0:
                        self.load_machine(machine)
                        self.n_wait[machine]=0
                        self.run_buffer(action)
                        self.processing[machine] = True
                        self.Tr[machine]=0
                        self.mp[machine][part] = self.parts[part][part]
                    else:
                        print('check if there is any part in machine and whether it is SB')
                else:
                    if not self.processing[machine] and not self.mready[machine] and self.n_wait[machine]==1 and self.Tr[machine]==0:
                        self.unload_machine(machine)
                        self.plus_buffer(action)
                        self.mp[machine]= [0,0,0]
                        self.unload_check[machine]= True
                        # if self.unloading[machine] == self.Tu[machine]:
                        # if machine==(n-1):
                        #     self.prod_count[machine][part] += parts[part, part]
                        self.load_machine(machine)
                        self.n_wait[machine] = 0
                        self.processing[machine]= True
                        self.waiting_time[machine]+= 1
                        if machine != 0:
                            self.minus_buffer(action)
                        else:
                            self.mp[machine][part] = 1
        elif (self.gs[machine] == 1) and not self.processing[machine] and not self.mready[machine] and self.n_wait[machine]==1 and self.Tr[machine]==0 and np.sum(self.mp[machine])==0:
            self.load_machine(machine)
            self.n_wait[machine] = 0
            self.processing[machine]= True
            if machine!=0:
                self.minus_buffer(action)
            else:
            # if np.all(self.b[machine - 1] >= self.parts[part]):
                self.mp[machine][part] = 1
        elif np.sum(gs) == ng:
            self.n_wait[machine]=1
            self.processing[machine] = False
            self.mready[machine] = False

        return self.gs, self.mp

    def minus_buffer(self, action):
        part, machine = action
        if np.all(self.mp[machine] == np.zeros(self.ptypes)):
            if np.all(self.b[machine - 1] >= self.parts[part]):
                # self.mp[machine, part] = 1
                self.b[machine - 1][part] -= 1
                self.mp[machine][part] = 1
            else:
                idxs = []
                sp = []
                for idx, val in enumerate(self.b[machine - 1]):
                    if val != 0:    #[0,0,0]
                        idxs.append(idx)
                        sp.append(val)
                if len(sp)>0:
                    selected_part= self.b[machine - 1].index(sp[0])
                    self.mp[machine][selected_part] = 1
                    self.b[machine - 1][selected_part] -= 1
                # else:
                #     self.n_SB[machine] = 1

    def plus_buffer(self, action):
        part, machine = action

        if np.any(self.mp[machine] > np.zeros(self.ptypes)) and machine!=(n-1):
            part_to_unload = list(self.mp[machine]).index(1)  # it turns out the index of the part already on the machine. The index also represents the part type
            part_to_unload_sequence = list(self.p_sequence[part_to_unload])  # let say [1,0,1,1]
            # for k in part_to_unload_sequence:  # maybe we can use np.where() function after this step
            if part_to_unload_sequence[machine + 1] == 1:  # maybe a loop can be used to check the next process of the part in sequence
                if self.B[machine] > np.sum(self.b[machine]):  # checking which next machine has an operation on this unloaded part
                    self.b[machine][part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    self.prod_count[machine, part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    # self.mp[machine, part] = 1  # new action: assign part j to machine i
                else:
                    self.n_SB[machine] = 1
            elif part_to_unload_sequence[machine + 2] == 1:
                if self.B[machine + 1] > np.sum(self.b[machine + 1]):  # checking which next machine has an operation on this unloaded part
                    self.b[machine + 1][part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    self.prod_count[machine, part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    # self.mp[machine, part] = 1
                else:
                    self.n_SB[machine] = 1
            elif part_to_unload_sequence[machine + 3] == 1:
                if self.B[machine + 2] > np.sum(self.b[machine + 2]):  # checking which next machine has an operation on this unloaded part
                    self.b[machine + 2][part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    self.prod_count[machine, part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    # self.mp[machine, part] = 1
                else:
                    self.n_SB[machine] = 1
        if machine == (n-1) and np.any(self.mp[machine] > np.zeros(self.ptypes)):
            part_to_unload = list(self.mp[machine]).index(1)
            self.prod_count[machine][part_to_unload]+=1
    def run_buffer(self, action):  # action: (part, mach)
        part, machine = action
        # gantry_status= self.Gantry(action)
        # if np.sum(gantry_status) < ng:
        if machine == 0:  # first machine
            self.plus_buffer(action)
        if machine != 0 and machine!=(n-1):  # In-between machines
            self.plus_buffer(action)
            self.minus_buffer(action)
        if machine==(self.n-1):    #last machine
            self.minus_buffer(action)

        return self.b, self.mp, self.prod_count  # we may take the parts with machine from the machine but i think it should be fine here as well...

    def reset(self):  # should we reset the buffers to zero and prod.count to zero
        self.Tr = [0] * self.n
        self.processing = [0] * self.n  # it is either True or False for each machine i.e. processing or not
        self.gs = [0] * self.n
        self.mp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        self.n_SB= [0] * self.n
        self.n_wait= [0] * self.n
        self.t=0

    def get_state(self):
        s = [self.ms[j] for j in range(n)] + [self.mp[j] for j in range(n)]
        a = tuple(s)
        mstates = tuple(y for x in a for y in (x if isinstance(x, list) else (x,)))
        return mstates

    def get_prod_count(self):
        self.m0_prod= []
        self.m1_prod = []
        self.m2_prod = []
        self.m3_prod = []
        self.m0_prod.append(self.prod_count[0])
        self.m1_prod.append(self.prod_count[1])
        self.m2_prod.append(self.prod_count[2])
        self.m3_prod.append(self.prod_count[3])
        return self.m0_prod, self.m1_prod, self.m2_prod, self.m3_prod

    def get_reward(self):
        cod = np.zeros(ptypes)
        cos = np.zeros(ptypes)
        # if self.ms[s_criticalIndx]==0:   # slowest critical machine index # reward needs to be changed to D/Tp[slowest_critical]
        ppl= self.downtime[s_criticalIndx]/(self.Tl[s_criticalIndx] + self.Tp[s_criticalIndx] + self.Tu[s_criticalIndx])
        reward=0
        for j in range(ptypes):
            if D[j]> self.prod_count[-1][j]:
                cod[j]= D[j]-self.prod_count[-1][j]
            else:
                cod[j]=0
        CoD= np.sum(omega_d * cod)

        for j in range(ptypes):
            if self.prod_count[-1][j]> D[j]:
                cos[j]= self.prod_count[-1][j]- D[j]
            else:
                cos[j]=0
        CoS= np.sum(omega_d * cos)
        CoT = CoD + CoS

        reward= -ppl-CoT
        return reward

    def get_info(self):
        return {"Prod_counts": self.prod_count[self.n - 1], "Gantry_state": self.gs, 'Machine_state': self.ms,
                'Parts_with_machine': self.mp}

    def get_W(self):
        for j in range(self.n):
            if self.W[j] == 0:
                if self.TBF[j] == 0:
                    self.W[j] = 1
                    self.ms[j] = 0   # down
                    self.TBF[j] = np.random.geometric(1 / self.MTBF[j])
                else:
                    self.TBF[j] -= 1
            if self.W[j] == 1:
                if self.TTR[j] == 0:
                    self.W[j] = 0
                    self.ms[j] = 1  # up
                    self.TTR[j] = np.random.geometric(1 / self.MTTR[j])
                else:
                    self.TTR[j] -= 1

    def get_terminated(self):
        self.terminated = self.t >= self.T
        return self.terminated

    def get_feature_size(self):
        # Return the size of your defined feature
        # Type: Integer
        feature_size = len(self.get_state())
        return feature_size

    # def action_space_n(self):
    #     return len(self.actionList)

    def get_action_size(self):
        return len(self.actionList)

    def get_feature_scale(self):
        return None

    def get_reward_scale(self):
        return None

    def mp_has_nullPart(self):
        # self.mp = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]])
        null_part = [0, 0, 0]
        result = []
        for i in self.mp:
            if np.all(i == null_part):
                result.append(1)
            else:
                result.append(0)
        if np.sum(result) == 0:
            test = False
        else:
            test = True
        return test

    def select_action(self,machine_index):
        feasible_actions = []
        for i in self.actionList:
            if i[1] == machine_index:
                feasible_actions.append(i)
        selected_action = random.choice(feasible_actions)
        # print(feasible_actions)
        return selected_action

    def machine_to_be_unloaded(self):
        machine_to_be_unload = None
        indxs_mprogress = []
        indxs_Tr = []
        for i, j in enumerate(self.mprogress):
            if j == 0:
                indxs_mprogress.append(i)
        # print(indxs_mprogress)
        for l, m in enumerate(self.Tr):
            if m == 0:
                indxs_Tr.append(l)
        # print(indxs_Tr)
        for i in indxs_mprogress:
            for j in indxs_Tr:
                # print(i,j)
                if i == j and self.processing[i] == False and self.mready[i] == False:
                    machine_to_be_unload = i
                # else:
                #     print('No machine is ready')
        return machine_to_be_unload


    def step(self):
        # part, machine = action

        if self.t == 0:
            self.n_SB= [0, 0, 0, 0]
            self.mp= np.zeros([n, ptypes])
            self.processing=[False, False, False, False]
            self.n_wait = [True, True, True, True]
            self.gs= [0, 0, 0, 0]
            self.ms= [1, 1, 1, 1]

        # print(self.t, self.T)
        # if self.ms[machine]==1 and not self.processing[machine] and self.n_SB[machine]==0 and not self.n_wait[machine] and np.sum(self.gs) <self.ng and all(self.mp[machine]==np.zeros(ptypes)):
        while self.t < self.T:
            [self.run_machine(k) for k in range(n)]
            if np.any(self.mp== [0,0,0]):
                action = random.choice(actions_list)
                part, machine = action
                if np.all(self.mp[machine]== [0,0,0]):
                    self.get_W()
                    # print('self.W[machine]=', self.W[machine])
                    if self.W[machine]==0:
                        self.run_gantry(action)
                        self.run_machine(machine)
                        # self.run_buffer(action)
                    else:
                        self.downtime[machine]+=1
                        self.total_downtime[machine]+=1
                else:
                    if self.mprogress[machine] == 0 and self.Tr[machine] == 0 and self.processing[machine] == False and self.n_wait[machine] == 1 and self.mready[machine] == False:
                        self.run_gantry(action)
                        self.run_machine(machine)
            else:
                if np.any(self.mprogress == 0) and np.any(self.Tr == 0) and np.any(self.processing == False) and \
                        np.any(self.n_wait == 1) and np.any(self.mready == False):
                    action = random.choice(actions_list)
                    part, machine = action
                    self.run_gantry(action)
                    self.run_machine(machine)
            reward= self.get_reward()
            # print('Timestep :',self.t, ', prod count: ',self.prod_count, ', buffer levels: ',self.b, ', reward: ', reward, ', parts with machine mp: ', self.mp,\
            #       ', machine current progress: ', self.mprogress, ', downtime, ', self.downtime, ', total downtime: ', self.total_downtime)
            self.t += 1

        return self.get_state(), self.get_reward(), self.get_terminated(), self.get_info()

if __name__ == "__main__":

    # from types import SimpleNamespace as SN
    # import yaml
    # from dqn import DDQN

    # with open('deepq.yaml', 'r') as f:
    # config = SN(**yaml.load(f, Loader=yaml.FullLoader))#

    env = Multiproduct(ptypes, n, b, B, Tp, ng, MTTR, MTBF, T, Tl, Tu)

    # print(env.get_alist())
    actions_list = env.get_alist()
    # print('mp: ', env.mp)
    for i in range(3):  # env.n
        action = random.choice(actions_list)
        # print('i=', i, 'action=', action)
        # print('env.step(action)=', env.step())
    #
    #
        s = env.reset()
        print("ep", i)
        # print(env.P[env.n - 1].X)
        # terminated = False

        step = 0

        while (t<T):
            step += 1
            t+=1
            print(env.get_state())
            # env.reset(0)
            print("STEP: {}".format(step))

            # print("state: {}".format(s))

            # avail_action = env.get_avail_action()
            # print("avail_action: {}".format(sum(avail_action)))
            # rand_action = [0] * env.action_space_n()

            # p = np.random.rand(env.action_space_n())

            # action = np.argmax(p)
            # action = 3
            actionslist= env.get_alist()
            action= random.choice(actionslist)

            # print("rand_action: {}".format(action))
            # rand_action[action] = 1
            # print(rand_action)
            s, reward, terminated, info = env.step()

            print(reward)
    #
