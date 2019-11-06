import pandas as pd
import matplotlib.pyplot as plt
import sys

#Receive from command line path to .csv file
datapath = sys.argv[1]

data = pd.read_csv(datapath + 'epochs_data.csv') 

#Recover experiment data
epochs = data['epoch']
train_losses = data['trainLoss']
train_loss_deviation = data['trainLossStandartDeviation']
train_accuracy = data['trainAccuracy']	
train_accuracy_deviation = data['trainAccuracyStandartDeviation']
validation_losses = data['validationLoss']
validation_loss_deviation = data['validationLossStandartDeviation']
validation_accuracy = data['validationAccuracy']
validation_accuracy_deviation = data['validationAccuracyStandartDeviation']


#Plot graphics

fig	, (ax1,ax2) = plt.subplots(2)

#Plot Loss X Epoch graphic
ax1.set_ylabel('Loss')
ax1.plot(epochs,train_losses,label='Train',color='C0')
ax1.fill_between(epochs,train_losses+train_loss_deviation,train_losses-train_loss_deviation, alpha=.3)
ax1.plot(epochs,validation_losses,label='Validation ',color='C1')
ax1.fill_between(epochs,validation_losses+validation_loss_deviation,validation_losses-validation_loss_deviation, alpha=.3)

#Plot Accuracy X Epoch graphic
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.plot(epochs,train_accuracy,label='Train',color='C0')
ax2.fill_between(epochs,train_accuracy+train_accuracy_deviation,train_accuracy-train_accuracy_deviation, alpha=.3)
ax2.plot(epochs,validation_accuracy,label='Validation',color='C1')
ax2.fill_between(epochs,validation_accuracy+validation_accuracy_deviation,validation_accuracy-validation_accuracy_deviation, alpha=.3)
plt.legend(loc='lower right')

for ax in (ax1,ax2):
    ax.label_outer()

fig.tight_layout()
plt.savefig(datapath + 'loss-accuracyXepoch.png')


'''
plt.ylabel('LOSS')
plt.xlabel('EPOCHS')
plt.xticks(range(0, max(epochs) + 1, 10))
plt.plot(epochs, train_losses, label='Training')
plt.legend()
plt.plot(epochs, validation_losses, label='Validation')
plt.legend()

#plt.show()
plt.savefig(datapath + 'lossXepoch_graph.png')


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