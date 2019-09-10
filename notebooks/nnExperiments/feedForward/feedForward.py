import torch
from models import *
import torchvision
import torchvision.transforms as transforms
import CatalinaLCDataSet as ds
from sklearn.metrics import precision_recall_fscore_support
import pdb
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)
if device == 'cuda':
    torch.cuda.set_device(1)

# Load Data
train = ds.dataSet('train',
                   transform=torchvision.transforms.Compose([
                       ds.ToTensor()
                   ]))
test = ds.dataSet('test',
                  transform=torchvision.transforms.Compose([
                      ds.ToTensor()
                  ]))

# Hyper-parameters
input_size = train.input_size()
hidden_size = 500
num_classes = 2
num_epochs = 100
batch_size = 1000
learning_rate = 0.01


# Data loaders

dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=True)

testLoader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=True)

# Fully connected neural network with one hidden layer




model = NeuralNet(input_size, hidden_size, num_classes).to(device)
model = model.double()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
# optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.989)


test_scores = []


# Train the mode
for epoch in range(num_epochs):
    running_loss = 0.0
    tot_correct = 0
    total = 0
    for i, sample_batched in enumerate(dataloader):

        # Move tensors to the configured device
        x = sample_batched['features']
        y = sample_batched['label']

        x = x.to(device)
        y = y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(x)

        _, predicted = torch.max(outputs.data, 1)

        total += y.size(0)

        tot_correct += (predicted == y).sum().item()

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*x.size(0)


    epochLoss =running_loss/len(dataloader)
    print("Epoch {} Loss {}".format(epoch,epochLoss))
    print("Epoch {} Acca {}".format(epoch, tot_correct/total))

    
    f= open("trainLoss.dat","a")
    f.write(str(epoch)+" "+ str(epochLoss)+'\n')
    f.close()

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        model = model.eval()
        correct = 0
        total = 0
        totPreds = np.array([])
        totLabels = np.array([])

        #iterate through test data
        for i, sample_batched in enumerate(testLoader):
            x = sample_batched['features']
            y = sample_batched['label']

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            _, predicted = torch.max(outputs.data, 1)

            

            totPreds = np.concatenate((totPreds, predicted.cpu().numpy()))
            totLabels = np.concatenate((totLabels, y.cpu().numpy()))
            # pdb.set_trace()

        precision, recall, fscore, coverage = precision_recall_fscore_support(
            totLabels, totPreds)
        print('Precision: {:.4} Recall:{:.4f} Fscore: {:.4f} Coverage: {}'.format(
            precision[1], recall[1], fscore[1], coverage[1]))

        f= open("testLoss.dat","a")
        f.write(str(epoch)+" " +str(precision[1])+" "+str(recall[1])+" "+str(fscore[1])+'\n')
        f.close()

    model = model.train()
# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
