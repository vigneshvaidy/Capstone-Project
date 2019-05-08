import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

mean_train_acc = []
mean_valid_acc = []

for i in range(20,600,20):

    metadata_path = '/home/xelese/CapstoneProject/metadata/dump_bl-20190325-141754-'+str(i)+'.pkl'
    metadata = np.load(metadata_path)
    accuracy_train = metadata['accuracy_train']
    accuracy_eval_valid = metadata['accuracy_eval_valid']
    mean_train_acc.append(np.average(accuracy_train))
    mean_valid_acc.append(np.average(accuracy_eval_valid))

print('train: ', mean_train_acc)
print('valid: ', mean_valid_acc)

plt.plot(mean_train_acc, label = "train value",linestyle='dashed', linewidth = 2,color = 'green',)
plt.plot(mean_valid_acc, label = "validation value", linewidth = 2,color = 'orange')

plt.xlabel('epochs/20')
# naming the y axis
plt.ylabel('Percentage')
plt.title('Average Precision')
plt.show()
plt.legend()
plt.savefig('Precision')