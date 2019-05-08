# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('PDF')
import numpy as np
import matplotlib.pyplot as plt

metadata_path = "/home/xelese/CapstoneProject/metadata/dump_bl-20190325-141754-560.pkl"
metadata = np.load(metadata_path)

acc_eval_valid = metadata['accuracy_eval_valid']
acc_train = metadata['accuracy_train']

plt.plot(acc_train, label = "train value", linewidth = 2,color = 'blue',)
plt.plot(acc_eval_valid, label = "validation value", linewidth = 2,color = 'red')

# naming the x axis
plt.xlabel('epochs')
# naming the y axis
plt.ylabel('Percentage of accuracy')
plt.title('Training data accuracy')
plt.show()
plt.legend()
plt.savefig('myfig')
