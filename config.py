import os

root_dir = '/Users/xavier/Programming/SiameseNetwork/data'
training_dir = root_dir+"/train"
training_csv = root_dir+"/train_data.csv"
testing_dir = root_dir+"/test"
testing_csv = root_dir+"/test_data.csv"
model_dir = root_dir+'model_ckpt'

batch_size = 32
epochs = 20
device = 'mps'

max_lr = 0.01
opt_func = 'Adam'