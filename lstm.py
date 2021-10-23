import glob
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from copy import deepcopy
import argparse
import consts as c
import pandas as pd
import ast
import logging
import utils

log = logging.getLogger('epip_lstm')
log.addHandler(logging.NullHandler())

logfmt = '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'    
logging.basicConfig(
    level=logging.DEBUG,
    format=logfmt, 
    datefmt='%Y-%m-%d %H:%M',
    handlers=[
        logging.FileHandler(filename=f'{c.TRAIN_OUT}/train2.log', mode='a'),
        logging.StreamHandler()
    ])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

input_size = 39     #20 amino acids, 4 ambiguous amino acids	
hidden_size = 30   #
num_layers = 2
num_classes = 1
#sequence_length = 
learning_rate = 0.005
batch_size = 64
num_epochs = 10
embedding_dim = 128

def plot(label_lst, predict_lst, name):

    fpr, tpr, thresholds = metrics.roc_curve(label_lst, predict_lst, )
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(c.TRAIN_OUT + f'/{name}')


class RNN(nn.Module):
    def __init__(self, name, input_size, hidden_size, num_layers, num_classes, embedding_dim):
        super(RNN, self).__init__()
        self.name = name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # embed = self.embedding(x.long())
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ =self.lstm(
            x, (h0, c0)
        )
        # out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        out = out.squeeze(-1)
        return out

    def learn(self, sequence, labels, mask):
        
        prediction = self.forward(sequence)

        criterion = torch.nn.BCEWithLogitsLoss(size_average=True, weight = mask)
        loss = criterion(prediction, labels)
        log.info("loss: {}".format(loss))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


    def evaluate(self, test_loader, name):
        label_lst, prediction_lst = [], []
        for sequence, labels, mask in test_loader:
            prediction = self.forward(sequence)
            prediction = torch.sigmoid(prediction)
            for pred, label, msk in zip(prediction, labels, mask):
                num = sum(msk.tolist()) 
                pred = pred.tolist()[:num] 
                label = label.tolist()[:num] 
                label_lst.extend(label)
                prediction_lst.extend(pred)
        sort_pred = deepcopy(prediction_lst)
        sort_pred.sort() 
        threshold = sort_pred[int(len(sort_pred)*0.9)]
        float2binary = lambda x:0 if x<threshold else 1
        binary_pred_lst = list(map(float2binary, prediction_lst))
        plot(label_lst, prediction_lst, name)
        log.info('roc_auc: {}, F1: {}, prauc: {}'.format(roc_auc_score(label_lst, prediction_lst), 
                f1_score(label_lst, binary_pred_lst), 
                average_precision_score(label_lst, binary_pred_lst)))

    def reward(self):
        pass
        
        

class dataset(Dataset):
	def __init__(self, data):
		self.sequences = [i[0] for i in data]
		self.labels = [i[1] for i in data]
		self.mask = [i[2] for i in data] 

	def __getitem__(self, index):
		return self.sequences[index], self.labels[index], self.mask[index]

	def __len__(self):
		return len(self.labels)


def onehot(idx, length):
	lst = [0 for i in range(length)]
	lst[idx] = 1
	return lst 

def zerohot(length):
	return [0 for i in range(length)]

# what is the maxlength here
def standardize_data(data, aa_lst, q_lst, maxlength = 300, X='Antigen'):
        features_i, sequences, labels, masks = utils.get_aa(data)
        SS = utils.get_SS(data)
        features_ii = utils.get_features(data)
        length = len(data)
        standard_data = []
        for i in range(length):
            seq = sequences[i]
            ssp = SS[i]
            feature_i = features_i[i]
            feature_ii = features_ii[i]
            label = labels[i]
            mask = masks[i]
            # log.info('seq len: {}, ssp len: {}, feature i len: {}, feature ii len: {}'.format(len(seq), len(ssp), len(feature_i), len(feature_ii)))
            sequence = [seq[j] + ssp[j] + feature_i[j] + feature_ii[j] for j in range(len(seq))]
            sequence += (maxlength-len(sequence)) * [zerohot(len(aa_lst)+len(ssp[0])+len(feature_i[0])+len(feature_ii[0]))]
            label += (maxlength-len(label)) * [0]
            mask += (maxlength-len(mask)) * [False]
            
            sequence, label, mask = sequence[:maxlength], label[:maxlength], mask[:maxlength]
            sequence, label, mask = torch.FloatTensor(sequence), torch.FloatTensor(label), torch.BoolTensor(mask)
            # log.info('seq len: {}, label len: {}, mask len: {}'.format(sequence.shape, label.shape, mask.shape))
            standard_data.append((sequence, label, mask))
            
        return standard_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=c.DATA_SEC)
    parser.add_argument('--name', default='IEDB')
    args = parser.parse_args()

    train_file = glob.glob(args.data + f'/{args.name}*train*.csv')
    valid_file = glob.glob(args.data + f'/{args.name}*valid*.csv')
    test_file = glob.glob(args.data + f'/{args.name}*test*.csv')
    log.info('train file: {}, test file: {}'.format(train_file, test_file))
    # train_data = pd.read_csv(train_file[0], converters={'Y':ast.literal_eval})
    # valid_data = pd.read_csv(valid_file[0], converters={'Y':ast.literal_eval})
    # test_data = pd.read_csv(test_file[0], converters={'Y':ast.literal_eval})
    # log.info('train data: {}'.format(train_data))
    train_data = pd.read_csv(train_file[0], converters={'q8_prob':ast.literal_eval,'q3_prob':ast.literal_eval, 'phi':ast.literal_eval, 'psi':ast.literal_eval, 'rsa':ast.literal_eval, 'asa':ast.literal_eval, 'Y':ast.literal_eval})
    valid_data = pd.read_csv(valid_file[0], converters={'q8_prob':ast.literal_eval,'q3_prob':ast.literal_eval, 'phi':ast.literal_eval, 'psi':ast.literal_eval, 'rsa':ast.literal_eval, 'asa':ast.literal_eval, 'Y':ast.literal_eval})
    test_data = pd.read_csv(test_file[0], converters={'q8_prob':ast.literal_eval,'q3_prob':ast.literal_eval, 'phi':ast.literal_eval, 'psi':ast.literal_eval, 'rsa':ast.literal_eval, 'asa':ast.literal_eval, 'Y':ast.literal_eval})

    train_data = train_data[train_data.Antigen.str.len() == train_data.seq.str.len()].reset_index()
    valid_data = valid_data[valid_data.Antigen.str.len() == valid_data.seq.str.len()].reset_index()
    test_data = test_data[test_data.Antigen.str.len() == test_data.seq.str.len()].reset_index()
    
    # exceptions
    e1 = train_data[train_data.Antigen.str.len() != train_data.seq.str.len()]
    e2 = valid_data[valid_data.Antigen.str.len() != valid_data.seq.str.len()]
    e3 = test_data[test_data.Antigen.str.len() != test_data.seq.str.len()]
    exceptions = pd.concat([e1, e2, e3])
    exceptions.to_csv(f'{c.DATA_SEC}/{args.name}_exceptions.csv')
    lst = train_data['Antigen'].tolist()
    maxlen = max([len(A) for A in lst])
    log.info('max len: {}'.format(maxlen))
    # maxlen=2000

    train_data_stand = standardize_data(train_data, c.AA_LIST, c.Q8, maxlength=maxlen)
    valid_data_stand = standardize_data(valid_data, c.AA_LIST, c.Q8, maxlength=maxlen)
    test_data_stand = standardize_data(test_data, c.AA_LIST, c.Q8, maxlength=maxlen)

    train_set = dataset(train_data_stand)
    valid_set = dataset(valid_data_stand)
    test_set = dataset(test_data_stand)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    model = RNN(name ='IEDB_Epitope', hidden_size=hidden_size, input_size=input_size, num_layers=num_layers, num_classes=num_classes, embedding_dim=embedding_dim)
    for ep in range(num_epochs):
        log.info('epoch: {}'.format(str(ep)))
        for sequence, labels, mask in train_loader:
            model.learn(sequence, labels, mask)
        model.evaluate(test_loader, name=f'{model.name}_{str(ep)}.png')


if __name__ == "__main__":
    main()



        
        
        







