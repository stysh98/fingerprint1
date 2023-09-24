from train.train_discriminative import *
from train.train_gaussian import *
from train.train_no_probabilistic import *
from train.train_gmm import *
from termcolor import colored

file_path = "data/Train.txt"
# print(colored('train gaussian started', 'black', 'on_light_cyan'))
# train_gaussian(file_path)

# print(colored('train discriminative started', 'black', 'on_light_cyan'))
# train_discriminative(file_path, False)
# train_discriminative(file_path, True)

# print(colored('train no probabilistic started', 'black', 'on_light_cyan'))
# train_SVM(file_path)

print(colored('train GMM started', 'black', 'on_light_cyan'))
train_gmm(file_path)