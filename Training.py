from typing import Sequence
from functools import partial
import random
import torch
import numpy as np
import random

# DO NOT CHANGE HERE
def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(13)

# Use this for getting x label
def rand_sequence(n_seqs: int, seq_len: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        yield [random.randint(0, 4) for _ in range(seq_len)]

# Use this for getting y label
def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        # note that seq is a string, not a list
        if dimer == "CG":
            cgs += 1
    return cgs

# Alphabet helpers   
alphabet = 'NACGT'
dna2int = { a: i for a, i in zip(alphabet, range(5))}
int2dna = { i: a for a, i in zip(alphabet, range(5))}

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

# we prepared two datasets for training and evaluation
# training data scale we set to 2048
# we test on 512

def prepare_data(num_samples=100):
    # prepared the training and test data
    # you need to call rand_sequence and count_cpgs here to create the dataset
    # step 1
    X_dna_seqs_train = list(rand_sequence(num_samples))
    """
    hint:
        1. You can check X_dna_seqs_train by print, the data is ids which is your training X 
        2. You first convert ids back to DNA sequence
        3. Then you run count_cpgs which will yield CGs counts - this will be the labels (Y)
    """
    #step2
    temp = list(map(lambda x: ''.join(intseq_to_dnaseq(x)), X_dna_seqs_train)) # use intseq_to_dnaseq here to convert ids back to DNA seqs
    #step3
    y_dna_seqs = list(map(count_cpgs, temp)) # use count_cpgs here to generate labels with temp generated in step2
    
    return X_dna_seqs_train, y_dna_seqs
    
train_x, train_y = prepare_data(2048)
test_x, test_y = prepare_data(512)

# some config
LSTM_HIDDEN = 128
LSTM_LAYER = 1
batch_size = 32
learning_rate = 0.1
epoch_num = 30

# create data loader
from torch.utils.data import DataLoader, TensorDataset

train_x_tensor = torch.tensor(train_x, dtype=torch.long)
train_y_tensor = torch.tensor(train_y, dtype=torch.float)
test_x_tensor = torch.tensor(test_x, dtype=torch.long)
test_y_tensor = torch.tensor(test_y, dtype=torch.float)

train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
test_dataset = TensorDataset(test_x_tensor, test_y_tensor)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self):
        super(CpGPredictor, self).__init__()
        # TODO complete model, you are free to add whatever layers you need here
        # We do need a lstm and a classifier layer here but you are free to implement them in your way
        self.lstm = torch.nn.LSTM(input_size=5, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYER, batch_first=True)
        self.classifier = torch.nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        # TODO complete forward function
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out[:, -1, :])
        return logits
    
# init model / loss function / optimizer etc.
model = CpGPredictor()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training (you can modify the code below)
t_loss = .0
model.train()
model.zero_grad()
for epoch in range(epoch_num):
    for batch in train_data_loader:
        x, y = batch
        x = torch.nn.functional.one_hot(x, num_classes=5).float()
        y = y.view(-1, 1)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        t_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {t_loss}')
    t_loss = .0

# eval (you can modify the code below)
model.eval()

res_gs = []
res_pred = []

with torch.no_grad():
    for batch in test_data_loader:
        x, y = batch
        x = torch.nn.functional.one_hot(x, num_classes=5).float()
        y = y.view(-1, 1)
        
        pred = model(x)
        res_gs.extend(y.numpy())
        res_pred.extend(pred.numpy())

# Calculate RMSE
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(res_gs, res_pred, squared=False)
print(f'RMSE: {rmse}')