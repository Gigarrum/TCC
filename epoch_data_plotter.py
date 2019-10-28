import pandas as pd
import matplotlib.pyplot as plt
import sys

datapath = sys.argv[1]

data = pd.read_csv(datapath + 'epochs_data.csv') 

epochs = data['epoch']
train_losses = data['trainLoss']
validation_losses = data['validationLoss']

plt.ylabel('LOSS')
plt.xlabel('EPOCHS')
plt.xticks(range(0, max(epochs) + 1, 10))
plt.plot(epochs, train_losses, label='Training')
plt.legend()
plt.plot(epochs, validation_losses, label='Validation')
plt.legend()

#plt.show()
plt.savefig(datapath + 'lossXepoch_graph.png')




'''
filepath = "experiments/TheNet-db=10%-batch=256-Adam-lr=0,0001-mm=0,9-epcs=100/epochs_data.csv"
train_loss_list = []
validation_loss_list = []
accuracy_list = []

with open(filepath, 'r') as f:
    line = f.readline()

    while line:

        #Remove \n character from string
        line = line[:-1]

        #Split relevant data on sub-strings
        data_texts = line.split(',')

        for i in range(0,len(data_texts)):

            data_texts[i] = data_texts


        for text in data_texts:
            
            data_texts = line.split(',')

            header = text.split('[')[0]

            print(header)

            #Return data sub-string value
            value = float([text.index('[')+len('['):text.index(']')])

            print(value)

        line = f.readline()
'''
