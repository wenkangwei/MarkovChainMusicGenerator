import numpy as np
import mdptoolbox.mdp as mdp
import mdptoolbox.example
# import  sys,os
import src.KalmanFilter as kf
# sys.path.append(os.path.abspath("./pysynth.py"))
from PySynth.pysynth import make_wav
from src.MDP import MDP,generate_RewardMatrix, generate_ProbMatrix
# Training set
samples =[]
# song 0
samples.append([["c",4],["d",4],["e",4],["f",4],
                ["g",4],["a",4],["b",4],["a",8],
                ["g",8],["f",8],["e",8],["d",8],
                ["c",8],["d",16],["e",16],["f",16],
                ["g",16],["a",16],["b",16],["b",2],
                ["a",2],["g",2],["f",2],["e",2],
                ["d",2],["c",2],["d",4],["e",4]])
# song 1
samples.append([["b",1],["a#",8],["a",8],["g#",8],
                ["g",1],["f#",8],["f",8],["e",8],
                ["d#",2],["d",8],["c#",1],["c",8],
                ["c#",2],["d",8],["d#",2],["e",8],
                ["f",1],["f#",8],["g",8],["g#",1],
                ["a",1],["a#",8],["b",8],["b",8],
                ["b",8],["g",8],["g",8],["d",8]])
# song 2
samples.append([["d",2],["d",4],["c",8],["d",4],
                ["d",4],["c",8],["d",2],["d",4],
                ["c",8],["d",2],["c",8],["d",4],
                ["d",4],["c",8],["c",8],["d",2],
                ["f",8],["g",8],["e",8],["g",8],
                ["d",16],["c",16],["d",16],["c",16],
                ["e",8],["e",8],["f",8],["g",8]])
# song 3
samples.append([["d",8],["d",8],["c",16],["d",8],
                ["d",8],["c",16],["d",8],["d",8],
                ["c",16],["d",8],["e",8],["d",8],
                ["d",8],["c",16],["c",8],["d",8],
                ["e",16],["f",16],["c",16],["d",16],
                ["c",8],["d",4],["d",4],["d",4],
                ["c",8],["d",8],["d",8],["d",8]])
# song 4
samples.append([["f",8],["f",8],["e",16],["f",8],
                ["f",8],["e",16],["f",8],["f",8],
                ["e",16],["f",4],["g",8],["a",8],
                ["a",4],["b",4],["b",4],["f",8],
                ["d",4],["d",4],["c",8],["d",8],
                ["d",4],["e",4],["e",8],["f",8],
                ["e",4],["e",4],["g",8],["g",8]])
# song 5
samples.append([["d",4],["d",4],["c",8],["d",4],
                ["d",4],["c",8],["d",4],["d",4],
                ["c",8],["d",4],["b",8],["d",8],
                ["d",4],["c",8],["d",16],["f",16],
                ["e",16],["a",8],["d",8],["b",8],
                ["d",4],["c",4],["d",4],["c",8],
                ["d",4],["d",4],["c",8],["d",4]])
# song 6
samples.append([["c",8],["d",8],["e",8],["f",8],
                ["g",8],["a",8],["b",8],["b",8],
                ["a",16],["g",16],["f",16],["e",16],
                ["d",16],["c",16],["c",8],["d",8],
                ["e",16],["f",16],["g",16],["a",16],
                ["b",16],["b",8],["b",8],["c",4]])








def generate_DataSeq(num=10,sample = 1 ,mode = "manual"):
    """
    It generates a sequence of decision in array format.
    Each decision in the sequence is a state tuple containing a  pitch vector  and a duration vector

    :param mode:
        "manual" mode: output the sequence pre-defined in the function
        "auto" mode: output the sequence automatically.
    :param num:
        The number of the decision in the sequence. Each decision contains a state vector and an action vector
    :return:
        a generated sequence of decision, a set of state and a set of action
    """
    import random
    global samples

    data_seq = []
    states = []
    pitch_states = ["a", "a#", "b", "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#"]
    duration_states = [1,2,4,8,16]
    # duration_states = [1,2,3,4,5,6,7,8,16]
    # state set in list format
    for s0 in pitch_states:
        for s1 in duration_states:
            states.append([s0,s1])
    actions = states.copy()
    # print("state:",states[0],states[1], "shape:",states.shape)
    if mode == "manual":
        data_seq = samples[sample]
        # for s0 in range(len(states)):
        #     data_seq.append(states[s0])
        #     for s1 in range(len(states)):
        #         data_seq.append(states[s1])
        pass
    elif mode == "auto":
        # generate music note randomly
        for i in range(num):
            pitch_ind = random.randint(0, len(pitch_states)-1)
            duration_ind = random.randint(0, len(duration_states)-1)
            data_seq.append([pitch_states[pitch_ind], int(duration_states[duration_ind])])
        pass
    else:
        raise ValueError("Don't recognize mode: %s"%(mode))
    return  data_seq, states, actions,pitch_states, duration_states

# def generate_PitchDurSeq(num, mode ="manual"):
#     import random
#     data_seq = []
#     states = []
#     pitch_states = ["a", "a#", "b", "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#"]
#     duration_states = [1,2,4,8,16]
#
#     if mode == "manual":
#         pitch_seq = ["a","a","a","c","c","d","b","b","d#"]
#         pass
#     elif mode == "auto":
#         for i in range(num):
#             pitch_ind = random.randint(0, len(pitch_states)-1)
#             duration_ind = random.randint(0, len(duration_states)-1)
#             data_seq.append([pitch_states[pitch_ind], int(duration_states[duration_ind])])
#         pass
#     else:
#         raise ValueError("Don't recognize mode: %s"%(mode))
#
#     return pitch_states, duration_states



def makeMusic(max_steps=100, policy=None, music_note=None,bad_note_cnt=3):
    if policy is None:
        raise ValueError("Policy can't be None!")
    if music_note  is None:
        raise ValueError(" Music Note can't be None!")

    step = 0
    Generated_note =[]
    state = np.random.randint(0,len(policy)-1)
    cnt =0

    while step <max_steps:
        # state = np.random.randint(0,len(music_note)-1)
        used_set = state
        state = policy[state]
        if used_set == state:
            cnt += 1
            if cnt >bad_note_cnt:
                print("Bad note!")
                i =0
                while state == used_set :
                    state = policy[i]
                    i+=1
                    # if all action in the policy are the same one
                    # it is an awful policy
                    if i >= len(policy):
                        print("Bad Policy! Rewrite your Policy!")
                        break
                        # return  None
                cnt =0
        # add music note
        Generated_note.append(music_note[state])
        step+=1
    return  Generated_note

def capture_syllable(seq):

    pass


def make_MusicSyllable(state_set, action_set, sample_seq, syllable_num =5, syllable_len =5):
    """
    It creates a list of syllable, policy with the number of "syllable_num"
    using markov chain method

    :param state_set:
    a state set of music in form of [pitch,duratoin]
    :param action_set:
    an  action set in form of [pitch, duration]
    :param sample_seq:
    basic music note sequence to generate music syllable
    :param syllable_num:
    the number of syllable
    :return:
    a list of policy and a list of syllable
    """
    policy_ls =[]
    syllable_ls = []
    iter =0
    data_seq = sample_seq.copy()
    # create n syllables
    while iter < syllable_num:
        iter += 1
        print("iterate:",iter)
        # Generate transition probabilities
        Psa = generate_ProbMatrix(state_set,action_set,data_seq)
        # print("Psa",Psa)
        # print("Psa:",Psa[np.nonzero(Psa.A != 0)])

        # Generate reward matrix
        rewardMatrix = generate_RewardMatrix(state_set=state_set,action_set=action_set,Psa=Psa,max_reward=5)
        # Construct MDP object
        # and run MDP and output policy
        mdp_obj = MDP(state_set,action_set,rewardMatrix,discount=0.001,toler=0.1,max_iter=1000,MDPTrials=data_seq,Psa= Psa)
        mdp_obj.run()
        for ind, state in enumerate(state_set):
            print("state:", state ,"->",state_set[mdp_obj.policy[ind]])

        # make a new syllable with steps of 4
        # using policies randomly
        policy = mdp_obj.policy_mat[np.random.randint(-3,-1)]
        syllable = makeMusic(syllable_len,policy,state_set)

        #add new syllable and policy to list
        # and guarantee they are not the same melody
        if syllable_ls.count(syllable)>0:
            iter -= 1
            temp,null1,null2,null3,null4 = generate_DataSeq(mode="manual",sample=np.random.randint(0,5))
            del null1,null2,null3,null4
            data_seq.extend(temp)
        else:
            policy_ls.append(policy)
            syllable_ls.append(syllable)

    return syllable_ls, policy_ls


def make_MusicSyllable_PitchDur(pitch_set, dur_set,pitch_seq,dur_seq):
    """
    It takes the training sequences of pitch and duration to train the mdp model
    and then generate a song using mdp
    :param pitch_set:
     a set containing all pitch states
    :param dur_set:
    a set   containing all duration states
    :param pitch_seq:
    a training sequence of pitch
    :param dur_seq:
    a training sequence of duratoin
    :return:
    a song
    """
    song_pitch =[]
    song_dur = []

    # Create probability matrix Psa
    pitch_P = generate_ProbMatrix(pitch_set,pitch_set,process=pitch_seq)
    dur_P = generate_ProbMatrix(dur_set,dur_set,process=dur_seq)
    # print("pitch P",pitch_P)
    # print("duration P",dur_P)

    # generate reward matrix
    pitch_reward = generate_RewardMatrix(state_set=pitch_set,action_set=pitch_set,Psa=pitch_P,max_reward=10)
    dur_reward = generate_RewardMatrix(state_set=dur_set,action_set=dur_set,Psa=dur_P,max_reward=10)
    # print("pitch reward:",pitch_reward)
    # print("duration reward:",dur_reward)

    # create mdp object
    pitch_mdp = MDP(S=pitch_set, A= pitch_set, reward= pitch_reward,discount= 1,toler= 0.1,max_iter=1000,Psa=pitch_P)
    dur_mdp = MDP(S=dur_set, A= dur_set, reward= dur_reward,discount= 1,toler= 0.1,max_iter=1000,Psa=dur_P)
    # run and train mdp
    pitch_mdp.run()
    dur_mdp.run()
    print("pitch optimal policy:",pitch_mdp.policy)
    print("duration optimal policy",dur_mdp.policy)

    # create music
    iter =0
    pitch_policy_cnt =len(pitch_set)
    dur_policy_cnt = len(dur_set)
    while iter <20:
        iter += 1
        # randomly select policy
        pitch_policy = pitch_mdp.policy_mat[np.random.randint(-pitch_policy_cnt,-1)]
        dur_policy = dur_mdp.policy_mat[np.random.randint(-dur_policy_cnt,-1)]

        # create song
        # remove bad pitch policy and write song
        ls = makeMusic(max_steps=5,policy=pitch_policy,music_note=pitch_set)
        if ls== None:
            pitch_mdp.policy_mat.remove(pitch_policy)
            if pitch_policy_cnt>1:
                pitch_policy_cnt -= 1
        else:
            song_pitch.extend(ls)

        # remove bad duration policy and write song
        ls = makeMusic(max_steps=5,policy=dur_policy,music_note=dur_set)
        if ls == None:
            dur_mdp.policy_mat.remove(dur_policy)
            if dur_policy_cnt >1:
                dur_policy_cnt -= 1
        else:
            song_dur.extend(ls)

    Song = [[song_pitch[i],song_dur[i]] for i in range(len(song_dur))]

    return Song

def make_MusicSyllable_Online(state_set =None, action_set= None, process=None,song_len =100, syllable_len=5):
    """
    Using greedy algorithm to train the mdp policy.
    In each iteration, re-calculate the prbability matrix of the previous song so that the mdp model consider
    the whole preious music decision it made and makes the whole song fit the main melody of the training music sequence.

    :param state_set:
    :param action_set:
    :param process:
    :param syllable_num:
    :param syllable_len:
    :return:
    """

    # calculate Psa
    Psa = generate_ProbMatrix(state_set=state_set, action_set=action_set,process=process)
    # calculate reward
    reward = generate_RewardMatrix(Psa=Psa,max_reward=20)
    # construct MDP object
    mdp_obj = MDP(S=state_set,A=action_set,reward= reward,discount=0.1,toler=0.1,max_iter=500,Psa=Psa)
    # loop for training MDP to make song online
    iter =0
    song = []
    policy_ls = []
    train_interval =8
    max_reward = 20
    while iter <(song_len+300)/train_interval:
        iter +=1
        print("iteration:",iter)
        # train MDP model online
        mdp_obj.run(method="PolicyIter")
        # mdp_obj.policyIteration(maxIter=1000)
        policy = mdp_obj.policy_mat[np.random.randint(-4,-1)]
        policy_ls.append(policy)
        syllables = makeMusic(max_steps=train_interval,policy=policy,music_note=state_set)
        song.extend(syllables)

        #update probability matrix
        # Note: don't clean the value matrix and let the mdp consider the whole sequence's value
        new_Psa = generate_ProbMatrix(state_set=state_set,action_set=action_set,process=song)
        print(" Psa updated")
        #update reward matrix
        new_reward = generate_RewardMatrix(Psa=Psa,max_reward=max_reward,mode="update",state_set=state_set,action_set=action_set,updated_Psa=new_Psa)
        mdp_obj.reward =new_reward

        #Consider whether changing the max reward will do better
        # max_reward += max_reward/train_interval

    for i in policy_ls:
        print("Policy list:",i)

    return song[-201:-1]

def MDP_Demo1(song =0, dir=""):
    """
    Demo 1:
    it considers that a state has 2 state variables: pitch and duration and trains a mdp model to learn how to play
    music
    :return:
    """
    # Generate sample sequence, state set and action set used to create syllables
    data, state_set, action_set,_,_ = generate_DataSeq(num=50,mode="manual",sample=song)
    data_seq = data.copy()
    print("Seq",data_seq)

    # Generate 10 syllables with steps of 4
    # syllable_ls: a list of available syllables
    #  policy_ls : policy of each syllable
    syllable_ls, policy_ls = make_MusicSyllable(state_set,action_set,data_seq, syllable_num=10, syllable_len =5)
    print("policy list:")
    for policy in policy_ls:
        print(policy)

    # Create a sample song randomly according to the created syllables and train MDP model
    process_len =30
    sample_song =[]
    for i in range(process_len):
        if len(syllable_ls) >1:
            index = np.random.randint(0,len(syllable_ls)-1)
        else:
            index =0
        sample_song.append(syllable_ls[index])

    # create and train mdp
    Psa = generate_ProbMatrix(state_set=syllable_ls,action_set=syllable_ls,process=sample_song)
    song_reward = generate_RewardMatrix(state_set=syllable_ls,action_set=syllable_ls,Psa=Psa)
    song_mdp =MDP(syllable_ls,syllable_ls,song_reward,discount=1,toler=0.1,max_iter=1000,Psa=Psa)
    song_mdp.run()

    # make song with length of 100 syllables. Each syllable has length of 5 notes
    song = []
    music_syll= []
    for i in range(10):
        # choosing policy of selecting syllables
        a = np.random.randint(-4,-1)
        policy = song_mdp.policy_mat[a]
        # length of song = 10 x 10 x 5 =500
        music_syll.extend(makeMusic(max_steps=10,policy=policy, music_note=syllable_ls))

    # write Song
    music = music_syll
    for i in range(len(music)):
        for note in music[i]:
            song.append(note)
    print("Song list:",song)

    print("Song Policy:",song_mdp.policy_mat)
    for i, syllable in enumerate(syllable_ls):
        make_wav(syllable,fn=dir+"syllable_{}.wav".format(i))

    make_wav(song,fn= "Demo1_Song.wav")

def MDP_Demo2(song= 0,dir =""):
    """
    Demo 2:
    It generates a sequence of music by using 2 mdp model to create pitch list and duration list individually
    and then combining two list into a song sequence
    """
    # generate data sequence and music note
    data_seq,_,_,pitch_set,dur_set = generate_DataSeq(num=100, mode="manual", sample=song)
    pitch_seq = [data_seq[i][0] for i in range(len(data_seq))]
    dur_seq = [data_seq[i][1] for i in range(len(data_seq))]
    print("pitch:",pitch_seq)
    print("pitch set:",pitch_set)
    print("duration:",dur_seq)
    print("duration set:",dur_set)

    # make song
    Song = make_MusicSyllable_PitchDur(pitch_set,dur_set,pitch_seq,dur_seq)
    print("Song:",Song)

    # write song to file
    make_wav(Song, fn = dir+"Demo2_Song.wav")

    pass



def MDP_Demo3(song = 0, dir=""):
    """
    Using one MDP model to generate songs online whlie MDP is being trained
    Parameters:
    ------------
    sample: the index number of the sample song in the song list for MDP training
    :return:
    """
    # Generate sample sequence, state set and action set used to create syllables
    data, state_set, action_set,_,_ = generate_DataSeq(num=50,mode="manual",sample=song)
    data_seq = data.copy()
    print("Seq",data_seq)
    #  policy_ls : policy of each syllable
    Song = make_MusicSyllable_Online(state_set=state_set,action_set=action_set,process=data_seq, song_len=200, syllable_len =5)
    print("Song:",Song)
    # write song to file
    make_wav(Song, fn =dir+"Demo3_Song{}.wav".format(song))

    pass


if __name__ == "__main__":
    # Write training samples to wav files
    # for i,sample in enumerate(samples):
    #     print("Writing sample %d"%(i))
    #     make_wav(sample,fn="sample{}.wav".format(i))

    # using one MDP model to make song, in which the state set and action set are lists
    # and each state and each action have the form of [pitch, duration]
    dir = "../songs/"
    MDP_Demo1(3,dir)


    # Using 2 MDP models to make song, in which pitch states and duration states are separared
    MDP_Demo2(3,dir)

    # Using a MDP model to generate songs online, which means it trains the MDP using the song pieces the
    # MDP've already generated while MDP is generating song pieces.
    for i in range(3,7):
        MDP_Demo3(i,dir)
    pass