#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import preprocess as pp
import pickle
import sklearn.metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report




if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses a GPU!')
else:
    device = torch.device('cpu')
    print('The code uses a CPU...')


# In[ ]:


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self,N,dim,layer_hidden,layer_output,gamma,alpha,beta):
        super(MolecularGraphNeuralNetwork,self).__init__()
        self.embed_fingerprint=nn.Embedding(N,dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_output)])
       #二分类
        self.W_property = nn.Linear(dim, 2)
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)        
        self.gamma = gamma
        self.beta=beta
        
    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices
    
    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + self.gamma*torch.matmul(matrix, hidden_vectors)
    
    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)
    
    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        Smiles,fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)
        
        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        return Smiles,molecular_vectors
    
    def mlp(self, vectors):
        """ regressor based on multilayer perceptron."""
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs=self.W_property(vectors)
        return outputs
    
    def WeightedFocalLoss(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-CE_loss)
        F_loss = at*(1-pt)**self.beta * CE_loss
        return F_loss.mean()

    def forward_regressor(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            Smiles,molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            correct_values=correct_values.view(len(data_batch[0]))
            #为了减小数据不平衡的影响，用focal_loss
            loss = self.WeightedFocalLoss(predicted_values, correct_values)
         #   loss = F.cross_entropy(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                Smiles,molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values=predicted_values.argmax(dim=1)
            correct_values=correct_values.view(correct_values.shape[0])
#            predicted_values = predicted_values.to('cpu').data.numpy()
#            correct_values = correct_values.to('cpu').data.numpy()
#            predicted_values = np.concatenate(predicted_values)
#            correct_values = np.concatenate(correct_values)
            return Smiles,predicted_values, correct_values


# In[ ]:


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


# In[ ]:


class Tester(object):
    def __init__(self, model):
        self.model = model
    def test_regressor(self, dataset):
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        total_correct = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            (Smiles,  predicted_values,correct_values) = self.model.forward_regressor(
                                               data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_values.to('cpu').data.numpy())
            Ys.append(predicted_values.to('cpu').data.numpy())
            total_correct+=torch.eq(predicted_values,correct_values).float().sum().item()
        SMILES = SMILES.strip().split()
        T, Y = map(str, np.concatenate(Ts)), map(str, np.concatenate(Ys))
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, T, Y)])
        ACC = total_correct / N  
        return ACC,predictions
    
    def P_R(self, dataset):
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        total_correct = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            (Smiles,  predicted_values,correct_values) = self.model.forward_regressor(
                                               data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_values.to('cpu').data.numpy())
            Ys.append(predicted_values.to('cpu').data.numpy())
        Ts=np.concatenate(Ts)
        Ys=np.concatenate(Ys)
        precision = precision_score(Ts, Ys)
        recall = recall_score(Ts, Ys)
        F1=f1_score(Ts, Ys, average='binary')
        bACC=sklearn.metrics.balanced_accuracy_score(Ts,Ys)
        return precision,recall,F1
    
    def save_ACCs(self, ACCs, filename):
        with open(filename, 'a') as f:
            f.write(ACCs + '\n')
    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


# In[ ]:


def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(123)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


# In[ ]:


def dump_dictionary(dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(dictionary), f)


# In[ ]:


#model_1（用最新的model1）
radius=1
dim=283
layer_hidden=6
layer_output=3
batch_train=261
batch_test=261
lr=2e-4
lr_decay=0.95
decay_interval=10
iteration=200
N=12000
alpha=0.28
beta=3.337
gamma=0.246
path='./'
dataname='ER'
dataset_all = pp.create_dataset('ER_train.txt',path,dataname)
dataset_train, dataset_test = split_dataset(dataset_all, 0.8)
dataset_test, dataset_dev = split_dataset(dataset_test, 0.5)
lr, lr_decay = map(float, [lr, lr_decay])
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses a GPU!')
else:
    device = torch.device('cpu')
    print('The code uses a CPU...')
model = MolecularGraphNeuralNetwork(
        N, dim, layer_hidden, layer_output,gamma,alpha,beta).to(device)
trainer = Trainer(model)
tester = Tester(model)


# In[ ]:


file_model=path+'modelgoodbefore'+'.h5'
model.load_state_dict(torch.load(file_model, map_location=torch.device('cpu')))
for para in model.W_fingerprint.parameters():
    para.requires_grad = False   
print(model)



