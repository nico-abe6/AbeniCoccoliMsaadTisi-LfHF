## Initialization

Create a conda enviroment with python 3.7 in order to do not create conflicts between different libraries:
```
conda create -n "name_of_virtenv" python=3.7
```
Activate the environment:
```
conda activate "name_of_virtenv"
```
Clone the repository:
```
git clone https://github.com/nico-abe6/AbeniCoccoliMsaadTisi-LfHF.git
```
Then install all the necessary libraries in the requirement file through the command:
```
pip install -r requirements.txt
```
After having installed the requirements is necessary to upgrade the library pyglet even if is not supported by gym=0.15.4, after the installation you'll see a conflict but everything will work fine.
```
pip install pyglet==1.5.11
```

# Learning from Human Preferences

run the file 


## Learning From Demonstration
# Play the game and registration of data

In the OpenAI gym folder launch the play.py file with the following command
```
python3 play.py CartPole-v1 --delay=50`-o data
```
where CartPole is the enviroment to play in, delay is to slow down the frames in order to make it more easier for a human to play, data is the destination folder where the vectors with states,actions... are going to be stored.

# Train the agent with DQN

Now make a copy of the CartPoleDemo.txt from data folder to the /AbeniCoccoliMsaadTisi-LfHF/DRL-using-PyTorch/DQNfromDemo/Test
rename it differently but in this case you have to open manually the CartPole.py and insert the name of the file in the line of code 24 where the TEST list is.

run CartPole.py
```
python CartPole.py
```
then it is going to save the reward learning list and plot the results, the learning curve and the test results.

# Add Human Preference
Copy the .npy array with the rewards from HumanPreference to the /AbeniCoccoliMsaadTisi-LfHF/DRL-using-PyTorch/DQNfromDemo/Test folder
open the plot_learning_curves.py script, insert the names of the files correctly, rewards from HP and from LfD and DQN
run the plot_learning_curves.py to see the 3 methods learning curves compared.








