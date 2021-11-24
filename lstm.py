import glob
import torch
from torch import nn, optim
from tqdm import tqdm
from tdc.single_pred import Epitope
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, accuracy_score
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score, accuracy_score
from scipy.spatial.distance import hamming
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
import time



log = logging.getLogger('epip_lstm')
log.addHandler(logging.NullHandler())

logfmt = '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'    


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

input_size = 39     #20 amino acids, 4 ambiguous amino acids	
hidden_size = 30   #
num_layers = 2
num_classes = 1
#sequence_length = 
learning_rate = 0.005
batch_size = 64
num_epochs = 20
embedding_dim = 128
gamma = 0.99

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
        self.gamma = gamma
        self.reward_episode = list()
        self.policy_history = list()
        self.reward_history = list()
        self.loss_history = list()

    def reset_episode(self):
        self.reward_episode = list()
        self.policy_history = list()
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)

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

    # def learn(self, sequence, labels, mask):
        
    #     prediction = self.forward(sequence)

    #     criterion = torch.nn.BCEWithLogitsLoss(size_average=True, weight = mask)
    #     loss = criterion(prediction, labels)
    #     log.info("loss: {}".format(loss))
    #     self.opt.zero_grad()
    #     loss.backward()
    #     self.opt.step()


    # def evaluate(self, test_loader, name):
    #     label_lst, prediction_lst = [], []
    #     for sequence, labels, mask in test_loader:
    #         prediction = self.forward(sequence)
    #         prediction = torch.sigmoid(prediction)
    #         for pred, label, msk in zip(prediction, labels, mask):
    #             num = sum(msk.tolist()) 
    #             pred = pred.tolist()[:num] 
    #             label = label.tolist()[:num] 
    #             label_lst.extend(label)
    #             prediction_lst.extend(pred)
    #     sort_pred = deepcopy(prediction_lst)
    #     sort_pred.sort() 
    #     threshold = sort_pred[int(len(sort_pred)*0.9)]
    #     float2binary = lambda x:0 if x<threshold else 1
    #     binary_pred_lst = list(map(float2binary, prediction_lst))
    #     plot(label_lst, prediction_lst, name)
    #     log.info('roc_auc: {}, F1: {}, prauc: {}'.format(roc_auc_score(label_lst, prediction_lst), 
    #             f1_score(label_lst, binary_pred_lst), 
    #             average_precision_score(label_lst, binary_pred_lst)))

    # def reward(self):
    #     pass
               

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

def reward(prediction, mask):
    rewards = []
    # ham = hamming(prediction, labels) * len(prediction)
    # turns =  0
    prev, curr = prediction[0], prediction[0]
    for i in range(len(prediction)):
        prev, curr = curr, prediction[i]
        if prev and curr and mask[i]:
            rewards.extend([1])
        elif not prev and not curr and not mask[i]:
            rewards.extend([1])
        else:
            rewards.extend([0])           

    return rewards

# - |\sum y_i - 15|
def sparse_reward(y_pred_bi):
    rewards = []
    for i in range(1, len(y_pred_bi) + 1):
        curr = y_pred_bi[:i]
        r = -abs(sum(curr) - 15) + 15
        # r = -abs(sum(curr) - 15)
        rewards.extend([r])
    return rewards

def customized_loss(y_pred, y, mask, reward_f):
    R = 0
    prediction_lst, label_lst, mask_lst, rewards = [], [], [], []
    y_pred_ = torch.sigmoid(y_pred)
    for pred, label, msk in zip(y_pred_, y, mask):
        num = sum(msk.tolist()) 
        pred = pred.tolist()[:num] 
        label = label.tolist()[:num] 
        mask_lst.extend(msk.tolist())
        label_lst.extend(label)
        prediction_lst.extend(pred)
    sort_pred = deepcopy(prediction_lst)
    sort_pred.sort() 
    threshold = sort_pred[int(len(sort_pred)*0.9)]
    float2binary = lambda x:0 if x<threshold else 1
    binary_pred_lst = list(map(float2binary, prediction_lst))
    rewards_episode = []
    if reward_f == 'sparse':
        rewards_episode = sparse_reward(binary_pred_lst)
    else:
        rewards_episode = reward(binary_pred_lst, mask_lst)
    for r in rewards_episode[::-1]:
        R = r + gamma * R
        rewards.insert(0,R)
    # rewards = rewards_episode
    rewards = torch.FloatTensor(rewards)
    log.info('rewards before standardization: {}'.format(rewards))
    rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))
    # rewards = rewards + 0.01
    rewards = rewards.detach()
    log.info('rewards: {}'.format(rewards))
    y_pred_ = torch.unsqueeze(torch.reshape(y_pred_, (-1,)), 1)
    loss = torch.sum(torch.mul(torch.log(y_pred_), rewards).mul(-1), -1)
    
    return loss



def train(model, train_loader, optimizer, epoch_num, rewardf, criterion=None):
    epoch_loss = 0

    model.train()

    for sequence, labels, mask in tqdm(train_loader, f"Epoch: {epoch_num}"):
        
        prediction = model(sequence)
        # y_pred = deepcopy(prediction)
        loss_r = None
        if rewardf == 'both':
            loss_1 = customized_loss(prediction, labels, mask, 'smooth')
            loss_2 = customized_loss(prediction, labels, mask, 'sparse')
            loss_r = loss_1 + loss_2
        else:
            loss_r = customized_loss(prediction, labels, mask, rewardf)
        criterion = torch.nn.BCEWithLogitsLoss(size_average=True, weight = mask)
        # smooth reward: 10*reward + 0.015

        log.info('cross entropy loss: {}, reward loss: {}, diff: {}'.format(criterion(prediction, labels), loss_r[0], criterion(prediction, labels)-abs(loss_r[0])))
        loss = criterion(prediction, labels) + loss_r[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        time.sleep(0.001)

    return epoch_loss/len(train_loader)

def train_wo_reward(model, train_loader, optimizer, epoch_num, criterion=None):
    epoch_loss = 0

    model.train()

    for sequence, labels, mask in tqdm(train_loader, f"Epoch: {epoch_num}"):
    #     prediction = self.forward(sequence)

    #     criterion = torch.nn.BCEWithLogitsLoss(size_average=True, weight = mask)
    #     loss = criterion(prediction, labels)
    #     log.info("loss: {}".format(loss))
    #     self.opt.zero_grad()
    #     loss.backward()
    #     self.opt.step()
        
        prediction = model(sequence)
        criterion = torch.nn.BCEWithLogitsLoss(size_average=True, weight=mask)
        loss = criterion(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # time.sleep(0.001)

    return epoch_loss/len(train_loader)

def validate(model, valid_loader, epoch_num):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for sequence, labels, mask in tqdm(valid_loader, f"Epoch: {epoch_num}"):
            prediction = model(sequence)
            criterion = torch.nn.BCEWithLogitsLoss(size_average=True, weight = mask)
            loss = criterion(prediction, labels)
            val_loss += loss

    return val_loss/len(valid_loader)



def evaluate(model, test_loader, name):
    label_lst, prediction_lst = [], []
    
    model.eval()

    with torch.no_grad():
        for sequence, labels, mask in test_loader:
            prediction = model(sequence)
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
        # plot(label_lst, prediction_lst, name)
    
    return label_lst, prediction_lst, binary_pred_lst

def standardize_data_w0feature(data, vocab_lst, maxlength = 300):
	length = len(data)
	standard_data = []
	for i in range(length):
		antigen = data['Antigen'][i]
		Y = data['Y'][i] 
		sequence = [onehot(vocab_lst.index(s), len(vocab_lst)) for s in antigen] 
		labels = [0 for i in range(len(antigen))]
		mask = [True for i in range(len(labels))] # labels and mask have the same length
		sequence += (maxlength-len(sequence)) * [zerohot(len(vocab_lst))] #pad to consistent length
		labels += (maxlength-len(labels)) * [0] 
		mask += (maxlength-len(mask)) * [False] # pad to maxlength
		for y in Y:
			labels[y] = 1 		
		sequence, labels, mask = sequence[:maxlength], labels[:maxlength], mask[:maxlength]
		sequence, labels, mask = torch.FloatTensor(sequence), torch.FloatTensor(labels), torch.BoolTensor(mask) 
		# print(sequence.shape, labels.shape, mask.shape)
        # sequence is 2D, labels and mask are 1D
		standard_data.append((sequence, labels, mask))
	return standard_data 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=c.DATA_SEC)
    parser.add_argument('--feature', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--reward', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--input_size', default=39, type=lambda x: int(x))
    parser.add_argument('--f', default='sparse')
    parser.add_argument('--name', default='IEDB_Jespersen')
    parser.add_argument('--log', default='train') # format: reward_maxlen
    parser.add_argument('--maxlen', type=int, default=2000)
    parser.add_argument('--seed', default=1)
    args = parser.parse_args()


    
    logging.basicConfig(
    level=logging.DEBUG,
    format=logfmt, 
    datefmt='%Y-%m-%d %H:%M',
    handlers=[
        logging.FileHandler(filename=f'{c.TRAIN_OUT}/{args.log}.log', mode='w+'),
        logging.StreamHandler()
    ])
    log.info('seed: '.format(args.seed))
    torch.manual_seed(args.seed)
    
    train_data_stand, valid_data_stand, test_data_stand = None, None, None
    input_size = args.input_size
    if not args.feature:
        data =  Epitope(name = 'IEDB_Jespersen')
        split = data.get_split()
        train_data = split['train']
        valid_data = split['valid']
        test_data = split['test']

        train_data_stand = standardize_data_w0feature(train_data, c.AA_LIST)
        valid_data_stand = standardize_data_w0feature(valid_data, c.AA_LIST)
        test_data_stand = standardize_data_w0feature(test_data, c.AA_LIST)
    else:
        maxlen = args.maxlen
        train_file = glob.glob(args.data + f'/{args.name}*train*.csv')
        valid_file = glob.glob(args.data + f'/{args.name}*valid*.csv')
        test_file = glob.glob(args.data + f'/{args.name}*test*.csv')
        log.info('train file: {}, test file: {}'.format(train_file, test_file))
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


        train_data_stand = standardize_data(train_data, c.AA_LIST, c.Q8, maxlength=maxlen)
        valid_data_stand = standardize_data(valid_data, c.AA_LIST, c.Q8, maxlength=maxlen)
        test_data_stand = standardize_data(test_data, c.AA_LIST, c.Q8, maxlength=maxlen)

    train_set = dataset(train_data_stand)
    valid_set = dataset(valid_data_stand)
    test_set = dataset(test_data_stand)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)



    model = RNN(name ='IEDB_Epitope', hidden_size=hidden_size, input_size=input_size, num_layers=num_layers, num_classes=num_classes, embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_valid_loss = float('inf')

    validloss = []
    max_roc, epoch = 0, 0

    for ep in range(num_epochs):
        log.info('epoch: {}'.format(str(ep+1)))
        train_loss = None
        if args.reward:
            train_loss = train(model, train_loader, optimizer, ep+1, args.f)
        else:
            log.info("train without reward")
            train_loss = train_wo_reward(model, train_loader, optimizer, ep+1)
        valid_loss = validate(model, valid_loader, ep+1)
        label_lst, prediction_lst, binary_pred_lst = evaluate(model, test_loader, name=f'{model.name}_{str(ep+1)}.png')
        label_lst_v, prediction_lst_v, binary_pred_lst_v = evaluate(model, valid_loader, name=f'{model.name}_{str(ep+1)}.png')
        # tqdm.write(f'''End of Epoch: {ep+1}  |  Train Loss: {train_loss:.3f}  |  roc_auc: {roc_auc_score(label_lst, prediction_lst):.3f}  |  F1: {f1_score(label_lst, binary_pred_lst, average='micro'):.3f}  |  prauc: {average_precision_score(label_lst, binary_pred_lst):.3f}''')
        
        log.info('Epoch: {}, loss: {}, valid loss: {}, roc_auc: {}, F1: {}, acc: {}, prauc: {}'.format(str(ep+1), train_loss, valid_loss, roc_auc_score(label_lst, prediction_lst), 
                f1_score(label_lst, binary_pred_lst), 
                accuracy_score(label_lst, binary_pred_lst),
                average_precision_score(label_lst, prediction_lst)))
        log.info('Validation set - Epoch: {}, loss: {}, valid loss: {}, roc_auc: {}, F1: {}, acc: {}, prauc: {}'.format(str(ep+1), train_loss, valid_loss, roc_auc_score(label_lst_v, prediction_lst_v), 
                f1_score(label_lst_v, binary_pred_lst_v), 
                accuracy_score(label_lst_v, binary_pred_lst_v),
                average_precision_score(label_lst_v, prediction_lst_v)))
        if max_roc < roc_auc_score(label_lst, prediction_lst):
            max_roc = roc_auc_score(label_lst, prediction_lst)
            epoch = ep
        validloss.extend([valid_loss])
        # if valid_loss <= min_valid_loss or valid_loss - min_valid_loss < 0.001:
        #     min_valid_loss = valid_loss
        # else:
        #     break
        # for sequence, labels, mask in train_loader:
        #     model.learn(sequence, labels, mask)
        # model.evaluate(test_loader, name=f'{model.name}_{str(ep)}.png')
    log.info('max roc_auc: {}, at epoch: {}, valid loss: {}'.format(max_roc, epoch, validloss))

if __name__ == "__main__":
    main()



        
        
        







