import pandas


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
