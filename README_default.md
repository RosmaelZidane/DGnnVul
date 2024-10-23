# Vuljava

# Project: This repository provides a code for training and validating a graph-based model for Java vul detection.

The running process is as follow:

git clone...

python3 -m venv .myvenv

source .myvenv/bin/activate

pip install -r requirement.py

sudo chmod +x getjoern.sh

./getjoern.sh

### we optain the pdg and pdg+raw graph

## The next challenge id to make the testing process of the training wotking. then reduice the 

# go and unzip external folder before starting


1- python3 prepareKB.py
2- python3 getgraph.py
3- python3 main_data_class.py
4- check file './storage/output/' to see relevant evaluation metrics.


'/home/rz.lekeufack/Rosmael/Vuljava/storage/'
/home/rz.lekeufack/Rosmael/Vuljava/storage/cache/

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# next thing to do, remove the code and obtain embedding from the description file

1- think about adding the cwe-id to the node data, then, during the learning process. you add a multiply  between a computed coeficient when that cwe-id is found
2- Try to idendify the line that is preditected as vulnerable. make a video of a clear function and its corresponding lines prediction

NB: in the final version that will be made public, make the runing process the way that after traing and collecting results for one step, delete old graph so that new ones can be build with for the new method we are evaluating.

If the experiment doesn't show clear difference, you can think about training and testing 3 to 5 time for each methods, then compare the average result, it will seems like k-flop validation.