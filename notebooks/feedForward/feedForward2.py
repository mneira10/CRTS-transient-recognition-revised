import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import CatalinaLCDataSet as ds


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
hidden_size1 = 500
hidden_size2 = 500
num_classes = 2
num_epochs = 50
batch_size = 100
learning_rate = 0.001


dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=True)

testLoader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=True)

# Fully connected neural network with one hidden layer


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


model = NeuralNet(input_size, hidden_size1,
                  hidden_size2, num_classes).to(device)
model = model.double()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
# optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.989)


# Train the mode
for epoch in range(num_epochs):
    running_loss = 0.0
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
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(x.size(0))
        print(loss.item())
        print(loss.item()*x.size(0))
        # print statistics
        running_loss += loss.item()*x.size(0)

    print("Epoch {} Loss {}".format(epoch, running_loss/len(dataloader)))

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        model = model.eval()
        correct = 0
        total = 0
        # for x, y in testLoader:
        for i, sample_batched in enumerate(testLoader):
            x = sample_batched['features']
            y = sample_batched['label']

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)

            correct += (predicted == y).sum().item()

        print('Acca test set: {:.2f} %'.format(100 * correct / total))
    
    model = model.train()
# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
