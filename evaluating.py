from evaluate.train_discriminative import *
from evaluate.train_gaussian import *
from evaluate.train_no_probabilistic import *
from evaluate.train_gmm import *
from termcolor import colored

file_path = "data/Train.txt"
file_path1 = "data/Test.txt"
# print(colored('train gaussian started', 'black', 'on_light_cyan'))
# train_gaussian(file_path, file_path1)

# print(colored('train discriminative started', 'black', 'on_light_cyan'))
# train_discriminative(file_path, False)
# train_discriminative(file_path, file_path1, True)

# print(colored('train no probabilistic started', 'black', 'on_light_cyan'))
# train_SVM(file_path, file_path1)

print(colored('train GMM started', 'black', 'on_light_cyan'))
train_gmm(file_path, file_path1)