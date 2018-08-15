import numpy as np
class MDP():
    def __init__(self, S, A,reward,discount, toler, max_iter, MDPTrials=None,Psa = None ):
        """

        :param S: the state set
            It contains all possible states in format [s0, s1, s2 . . .], where si is the state variable
        :param A: the action set
            It contains all possible actions in format [a0, a1, a2 . . .], where ai is the action variable

        :param Psa: transition probability matrix
            The transition probability matrix between states and actions

        :param Reward: reward matrix R(s)
            The reward matrix used to calculate the value
        :param discount:
            discount variable y
        :param toler: tolerance
            The tolerance of error. Once the change of value is less than toler, it is considered as converged
        :param max_iter:
            The maximum iteration times to iterate value function and policy


        members:
        S: state set with the size of m

        A: action set with the size of n

        discount: discount constant with value less than 1 and greater than 0

        toler: tolerance with value less than 1 and greater than 0

        max_iter: maximum iteration times

        reward: reward matrix in shape of m x n, where m is the length of state set and n is the length of action set

        V: the optimal value vector of the optimal policy in shape of m, where m is the length of actions
        V_mat: the optimal value matrix
        policy: policy mapping vector

        m: the number of states

        n: the number of actions
        """

        if S is None:
            raise  ValueError("State set can not be empty!")
        self.S = S

        if A is None:
            raise  ValueError("Action set can not be empty")
        self.A = A
        self.m = len(self.S)
        self.n =len(self.A)

        if reward is None:
            raise  ValueError("Reward matrix can not be empty")
        if Psa is None:
            if MDPTrials is None:
                raise  ValueError("Probability matrix can not be empty")
            else:
                self.Psa = self._calc_ProbabilityMatrix( MDPTrials)
        else:
            self.Psa =Psa

        if np.shape(reward) in ((len(S),),(1,len(S))):
            raise ValueError("Shape of probability matrix is wrong")


        self.reward = reward
        self.discount = discount
        self.toler = toler
        self.max_iter =max_iter
        self.V =np.zeros(self.n)
        self.V_mat = np.mat(np.zeros(np.shape(reward)))
        self.policy= np.arange(0,self.m)
        self.policy_mat = []

    def run(self, method= "ValueIter"):
        """
        Train the policy and value to get the optimal value matrix/vector
        and the optimal policy matrix/ vector

        :return:
            trained policy
        """
        import time
        # start
        begin_time = time.time()
        print("Using method:",method.lower())
        if method.lower() == "valueiter":
            # Value Iteration
            self.valueIteration(maxIter=self.max_iter)
        elif method.lower() == "policyiter":
            # policy iteration
            self.policyIteration(maxIter=self.max_iter)
        else:
            # Bellman operator
            self.V, self.policy = self._BellmanOperator(self.policy)
        end_time = time.time()
        print("Used time:", end_time - begin_time)
        pass

    def _calc_ProbabilityMatrix(self,trials):
        """
        Calculate the transition probability matrix Psa based on
        the trial given in initialization.

        :return:

        """
        Psa = np.mat(np.ones((self.m,self.n)))
        for s_index in range(len(trials)):
            if s_index+1 < len(trials):
                a_index = s_index +1
                # convert state and action to indices in probabiilty matrix
                s_pos = list(self.S).index(list(trials[s_index]))
                a_pos = list(self.A).index(list(trials[a_index]))
                Psa[s_pos,a_pos] += 1

        Psa /=  np.sum(Psa,axis=1)
        return Psa

    def _BellmanOperator(self,policy,V=None):
        """
        Bellman Operator for maximizing the value V(s)
        V(s) = R(s) + max(a): gamma *sum(Psa(s) * V(s))
        :return:
        V : new value vector with the given policy
        """
        # if state is not int:
        #     raise ValueError("Input state must be index")
        # if action is not int:
        #     raise ValueError("Input action must be index")

        if V is None:
            V = self.V

        # method 1: partial element-wise
        FutureReward_mat = np.mat(np.multiply(self.Psa,V))
        FutureReward = np.sum(np.max(FutureReward_mat,axis=1))
        for state in range(self.m):
            action = policy[state]
            V[action] = self.reward[state,action] + self.discount * FutureReward
            self.V_mat[state,action] = V[action]

        # Reduce dimension of policy list
        # Note: np.argmax(axis=0):
        #   it finds the maximum value's index in each col and then return a index sequence in shape of 1 x m, where m = # of cols
        # np.argmax(axis=1):
        #   it finds the maximum value's index in each row along vertical axis. Then return n x1 index sequence, where n = # of rows

        # update policy matrix
        weighed_V =np.multiply(self.Psa,V)
        temp = np.zeros((self.m,self.n))

        # For each state, sort the value in ascent order and then
        # return the sorted index to policy. The last index is the best action's index
        for row_index, row in enumerate(weighed_V):
            temp[row_index,:] = np.argsort(row)

        # Note: some values of different action at the same state may be same, which
        # means it is possible that taking the optimal policy will lead the the same maximum value
        #  as the second or even the third optimal policy does. To avoid making the same decision all
        #  the time, we can choose the first or second or even third policy randomly
        self.policy_mat = temp.transpose().astype(int).tolist()
        # print("policy mat",np.mat(self.policy_mat))

        new_policy = self.policy_mat[-1]
        # new_policy = np.argmax(weighed_V,axis=1).transpose().tolist()[0]
        # print("Policy:",new_policy)
        return V, np.array(new_policy)


    def policyIteration(self,maxIter = 1):
        """
        Policy iteration to update policy
        :return:
        """
        iter =0
        while iter < maxIter:
            iter += 1
            pre_policy = self.policy
            # update value vector and policy
            _, self.policy = self._BellmanOperator(self.policy)
            # check if policy converge
            diff = np.abs(self.policy - pre_policy)
            cnt = 0
            for d in diff:
                if d > self.toler:
                    cnt+=1
            if cnt == 0:
                break
            pass
        # if policy converges, return the eventual policy
        return  self.policy

    def valueIteration(self,maxIter=1):
        """
        Value iteration using Bellman operator

        :return:
        """
        pre_V = self.V
        iter = 0
        policy =self.policy
        # loop through and update the value vector
        while iter < maxIter:
            iter += 1
            # update value vector
            for s_index, _ in enumerate(self.S):
                self.V, policy = self._BellmanOperator(policy)

            # check if value function/vector converges and diff < tolerance
            diff = abs(self.V - pre_V)
            cnt =0
            for d in diff:
                if d >= self.toler:
                    cnt += 1
            if cnt == 0:
                break

            iter+=1
        # if the value matrix / value function converges, return its final policy as the optimal policy
        self.policy = policy
        return self.policy



def calc_Probability(state, action, process):
    """
    Calculate the probability of taking action "action" at state "state" using element-wise method
    :param state:
    The current state
    :param action:
    The action token at the current state
    :param process:
    The process used to train the MDP model
    :return:
    Probability of taking this action at current state P = action_count_at_this_state / total # of this state

    """
    P =0.0
    s_cnt =0.0
    a_cnt = 0.0
    for s_index in range(len(process)):
        if s_index+1 < len(process):
            a_index = s_index+1
            if state[0] == process[s_index][0] and state[1] == process[s_index][1]:
                s_cnt += 1
                if action[0] == process[a_index][0] and action[1] == process[a_index][1]:
                    a_cnt += 1
        else:
            pass
    if s_cnt != 0:
        P = a_cnt / s_cnt
    return P

def generate_ProbMatrix(state_set = None, action_set=None, process=None):
    """
    Generate the transition probability matrix Psa
    according to the given Markov decision trial
    :param
    --------
    state_set: the set of all states, which should be in array
    action_set: the set of all actions
    prccess : the decision sequence. It should be an array or a list here
    :return:
        m x m probability matrix.
        Each row represents a state. Each col represents an action
    """

    # check if input are valid
    if state_set is None:
        raise ValueError("State set can't be empty")
    if action_set is None:
        raise ValueError("Action set can't be empty")
    if process is None:
        raise ValueError("Sample process can't be empty")


    # calculate transition probability matrix
    states_num = len(state_set)
    action_num = len(action_set)

    # create smoothed probability matrix
    Psa = np.mat(np.ones([states_num,states_num])) / action_num
    # method 1: element-wise without smoothing
    # for s_index, state in enumerate(state_set):
    #     for a_index, action in enumerate(action_set):
    #         loop through the process
    # P = calc_Probability(state,action, process)
    # Psa[s_index,a_index] =P
    # pass

    # method 2: using smoothing
    for s_index in range(len(process)):
        if s_index +1 < len(process):
            a_index = s_index +1
            # return indices of state and action
            try:
                state_ind = state_set.index(process[s_index])
                action_ind = action_set.index(process[a_index])
                Psa[state_ind,action_ind ] +=1
            except:
                raise ValueError("Process data doesn't match state or action set:",process[s_index],process[a_index])
                pass

    index = np.nonzero(np.sum(Psa,axis=1) >0)[0]
    Psa[index] = np.divide(Psa[index], np.sum(Psa,axis=1)[index])
    return  Psa


def generate_RewardMatrix(Psa, max_reward=10.0, mode = "stationary", state_set = None, action_set = None, updated_Psa =None):
    """
    It generate the reward matrix in shape mxm, where m is the total number of the states
    :param state_set:
        state set in array format [s0,s1,. . .], where si is a set containing state variables
        variable si
    :param action_set:
        action set in array format [a0,a1,a2 . . .], where ai is a set containing action variables
        variable ai
    :param Psa:
        the transition probability matrix used to generate the reward matrix
    :param max_reward:
        the maximum value the reward can reach
    :return:
        reward matrix in shape m x m

    """

    Reward = np.mat(Psa.dot(max_reward))
    if mode.lower() == "stationary":
        # Reward += np.mat(np.ones(np.shape(Psa))) * (1/len(Psa))
        pass
    elif mode.lower() == "update":
        if np.shape(updated_Psa) != np.shape(Psa):
            raise  ValueError("Two Psa matrix must be in the same shape! Psa:",Psa.shape,"new Psa:",np.shape(updated_Psa))
        else:
            diff_reward = (Psa - updated_Psa) * max_reward
            Reward -= diff_reward

            pass
    else:
        raise ValueError("Can't recognize mode:",mode)

    return  Reward

