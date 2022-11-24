## Initialization

create a conda enviroment in order to do not create conficts between different libraries, with python 3.7 and all the packages in the requirements.txt.

*conda create -n "name_of_virtenv" python=3.7

*conda activate "name_of_virtenv"

*pip install -r reqiurements.txt


# Learning from Human Preferences

run the file 


## Learning From Demonstration
# Play the game and registration of data

In the OpenAI gym folder launch the play.py file with the following command

*python3 play.py CartPole-v1 --delay=50`-o data

where CartPole is the enviroment to play in, delay is slow down the frames in order to make it more easier for a human to play, data is the destination folder where the vectors with states,actions... are going to be stored.


# Train the agent with DQN

Now make a copy of the CartPoleDemo.txt from data folder to the /AbeniCoccoliMsaadTisi-LfHF/DRL-using-PyTorch/DQNfromDemo/Test
rename it differently but in this case you have to open manually the CartPole.py and insert the name of the file manually in the line of code 24 whre the TEST list is.

run CartPole.py

*python CartPole.py

then it is going to save the reward learning list and plot the results, the learning curve and the test results.

# Add Human Preference

copy the .npy array with the rewards from HumanPreference to the /AbeniCoccoliMsaadTisi-LfHF/DRL-using-PyTorch/DQNfromDemo/Test folder
open the grafico.py script, insert the names of the files correctly, rewards from HP and from LfD and DQN
run the grafico.py to see the 3 methods learning curves compared.








