import glob
import os
import random

# glob.glob('/raw——data/data_test/*.csv')
path = 'C:/Users/Leeld/Documents/dataset/raw_data/'

with open('one_result.csv', 'w') as f:
    f.write('sample_file_name,label\n')
    for name in os.listdir(path + 'data_test'):
        # f.write(name + "," + str(random.randint(0, 1)) + '\n')
        f.write(name + ",1" + '\n')
