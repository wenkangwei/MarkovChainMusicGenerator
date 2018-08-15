---


---

<h3 id="introduction">Introduction</h3>
<p>This project is a sample example of using Markov decision process to generate songs. In this project, there are 3 approaches used to generate the song.</p>
<p>The basic idea of first method is generate  a number of melodies using the given samples before creating a full song first and then connect several melodies as a song. The selection and order of the melodies are obtained by a random sequence. In this method, only 1 MDP model is used and each state consists of a pitch state variable and duration variable.</p>
<p>The second method is to separate the pitch and duration and use 2 MDP models to train them respectively. That is to say, there are two state sets, the pitch state set and the duration state set. After training 2 models, the generated sequences of pitch and duration will be combined in order.</p>
<p>In the last demo, a online training method is used. The first  segment of song is generated by training the MDP using the given training sample.  Then the generated song pieces will be used as a new training sample to update the probability matrix. The reward matrix is updated by subtracting the difference between the original probability matrix and the current one, which can avoid the model always creates the same melodies and allow the song’s melodies to vary.</p>
<p>Among those three demos, the third one is better than the other. However, the performance of the demo is not always satisfying. Sometimes the main melody in the training sample becomes the major melody of the song and repeats continuously.</p>
<p>In my prospective, the update rule for the reward function / matrix could be the key to optimize the music generator. This project still need improvement in many places.</p>
<p>If you are interested in this project or have any ideas about improving this project, please feel free to commit here or contact me by <a href="mailto:wenkanw@g.clemson.edu">wenkanw@g.clemson.edu</a></p>
<h3 id="further-reading">Further Reading</h3>
<p><a href="https://cs.uwaterloo.ca/~klarson/teaching/F08-886/16MDP.pdf">Markov Decision Process</a><br>
<a href="https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-825-techniques-in-artificial-intelligence-sma-5504-fall-2002/lecture-notes/Lecture20FinalPart1.pdf">MDP</a></p>

