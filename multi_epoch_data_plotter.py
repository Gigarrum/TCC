import pandas as pd
import matplotlib.pyplot as plt
import sys

from itertools import permutations
from random import sample
import numpy as np


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



#Get cmatplotlib color cycle
#prop_cycle = plt.rcParams['axes.prop_cycle']
#colors = prop_cycle.by_key()['color']
cmap = get_cmap(len(sys.argv)-1)
#Receive from command line path to .csv files

fig	, (ax1,ax2) = plt.subplots(2,figsize=(10,5))


for i in range(1,len(sys.argv),2):

	data_i = i
	name_i = i + 1

	datapath = sys.argv[data_i]
	net_name = sys.argv[name_i]

	datapath.split('-')

	data = pd.read_csv(datapath + '/epochs_data.csv') 

	#Recover experiment data
	epochs = data['epoch']
	train_losses = data['trainLoss']
	#train_loss_deviation = data['trainLossStandartDeviation']
	train_accuracy = data['trainAccuracy']	
	#train_accuracy_deviation = data['trainAccuracyStandartDeviation']
	validation_losses = data['validationLoss']
	#validation_loss_deviation = data['validationLossStandartDeviation']
	validation_accuracy = data['validationAccuracy']
	#validation_accuracy_deviation = data['validationAccuracyStandartDeviation']

	#Plot graphics

	#Plot Loss X Epoch graphic
	ax1.set_ylabel('Loss')
	ax1.plot(epochs,train_losses,label='Treino ' + net_name,color=cmap(i))
	#ax1.fill_between(epochs,train_losses+train_loss_deviation,train_losses-train_loss_deviation, alpha=.1,color=cmap(i))
	ax1.plot(epochs,validation_losses,label='Validação ' + net_name,linestyle=':',color=cmap(i))
	#ax1.fill_between(epochs,validation_losses+validation_loss_deviation,validation_losses-validation_loss_deviation, alpha=.1,color=cmap(i))

	#Plot Accuracy X Epoch graphic
	ax2.set_xlabel('Épocas')
	ax2.set_ylabel('Acurácia')
	ax2.plot(epochs,train_accuracy,label='Treino ' + net_name,color=cmap(i))
	#ax2.fill_between(epochs,train_accuracy+train_accuracy_deviation,train_accuracy-train_accuracy_deviation, alpha=.1,color=cmap(i))
	ax2.plot(epochs,validation_accuracy,label='Validação ' + net_name,linestyle=':',color=cmap(i))
	#ax2.fill_between(epochs,validation_accuracy+validation_accuracy_deviation,validation_accuracy-validation_accuracy_deviation, alpha=.1,color=cmap(i))
	ax2.legend(bbox_to_anchor=(1.05,0.7))

	# Put a legend to the right of the current axis
	#fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	for ax in (ax1,ax2):
	    ax.label_outer()

fig.tight_layout()

#Save png for easy visualization
plt.savefig('multi-loss-accuracyXepoch.png')
#Save eps for vectorizade image
plt.savefig('multi-loss-accuracyXepoch.eps')
