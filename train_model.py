import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

num_epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv('cleaned_h1b_data_subset.csv')
df = df.drop(columns=['Unnamed: 0'])
print(df.info())

numerical_columns = ['PREVAILING_WAGE', 'YEAR', 'lon', 'lat']
df = df[df['CASE_STATUS'].isin(['CERTIFIED', 'DENIED'])]
df['target'] = (df['CASE_STATUS'] == 'CERTIFIED').astype(float)

X = df[numerical_columns].copy()
y = df['target']

Xt = torch.tensor(X.values, dtype=torch.float32)
yt = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dset = MyDataset(Xt, yt)
loader = DataLoader(dset, batch_size=24, shuffle=True)

#binary classification model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

input_size = len(numerical_columns)
model = SimpleClassifier(input_size)
loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model = model.to(device)

# for epoch in range(num_epochs):
#     epoch_loss = 0
#     num_batches = 0
    
#     for x, y in loader:
#         x = x.to(device)
#         y = y.to(device)
#         output = model(x)
#         loss = loss_func(output, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         epoch_loss += loss.item()
#         num_batches += 1
    
#     avg_loss = epoch_loss / num_batches
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# torch.save(model.state_dict(), 'h1b_model.pt')
# print("Model saved")

def get_model():
    model = SimpleClassifier(input_size)
    model.load_state_dict(torch.load('h1b_model.pt', map_location=device))
    model.eval()
    return model