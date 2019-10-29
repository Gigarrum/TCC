import pandas as pd
import matplotlib.pyplot as plt
import sys

#Receive from command line path to .csv file
datapath = sys.argv[1]

data = pd.read_csv(datapath + 'epochs_data.csv') 

epochs = data['epoch']
train_losses = data['trainLoss']
validation_losses = data['validationLoss']
validation_accuracy = data['accuracy']

plt.ylabel('LOSS')
plt.xlabel('EPOCHS')
plt.xticks(range(0, max(epochs) + 1, 10))
plt.plot(epochs, train_losses, label='Training')
plt.legend()
plt.plot(epochs, validation_losses, label='Validation')
plt.legend()

#plt.show()
plt.savefig(datapath + 'lossXepoch_graph.png')