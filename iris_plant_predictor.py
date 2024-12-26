import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# Neural network class
class Network(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.output = nn.Linear(h2_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, X):
        X = self.activation(self.fc1(X))
        X = self.activation(self.fc2(X))
        X = self.output(X)

        return X

# Custome data loader
class IrisDataSet(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # Remove extra white spaces and \n character
        parts = [line.strip().split(",") for line in lines if line.strip() and len(line.strip().split(",")) == 5]
        features = [list(map(float, part[:4])) for part in parts]
        labels = [part[4] for part in parts]
        self.data = [(feature, label) for feature, label in zip(features, labels)]
        
        self.mapped_labels = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2
        }
    
    def __getitem__(self, index):
        sample = self.data[index]
        features = sample[0]
        labels = sample[1]

        map_labels = self.mapped_labels[labels]

        # Convert feature ad labels to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        map_lables_tensor = torch.tensor(map_labels, dtype=torch.long)

        return features_tensor,map_lables_tensor

    def __len__(self):
        return len(self.data)

# Intantiate Model paramers
model = Network(input_size= 4, h1_size= 10, h2_size= 10, output_size= 3)

filepath = "/home/oem/Desktop/Academic/Python Projects/Iris plant classifier/iris.data"
dataset_ = IrisDataSet(filepath)

dataloader_ = DataLoader(dataset_,batch_size= 32, shuffle= True)

# Loss function
loss_func = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr= 0.001)

# Training loop
Epoch = 200
loss_values = []
for i in range(Epoch):
    total_loss = 0
    for feature, label in dataloader_:
        predictor = model(feature)
        loss = loss_func(predictor, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    
    loss_values.append(total_loss)

    print(f"Epoch {i+1}/{Epoch}, Loss: {total_loss:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(range(1, Epoch + 1), loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()



