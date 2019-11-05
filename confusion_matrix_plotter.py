import matplotlib.pyplot as plt
from DatasetAPPM import DatasetAPPM
from Nets import TheNet
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == "__main__":

  #Define device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  #Define pre-training data transformations
  transform = transforms.Compose(
    [transforms.ToTensor()])

  #Load Train set for inference
  print('Loading trainset...')

  trainset = DatasetAPPM(sys.argv[1],transform = transform)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                           shuffle=False, num_workers=0)
  print('Validationset loaded.')

  #Load Net model
  model = TheNet()
  model.load_state_dict(torch.load(sys.argv[2] + 'weights.pth'))

  print('Neural network evaluation mode activated!!!')
  model.eval()

  with torch.no_grad():
      for data in trainloader:
          grids, labels, resIdx = data
          print(labels)
          output = model(grids)
          print(output)
          APPM.append((resIdx,output))
          _, predicted = torch.max(output.data, 1)


