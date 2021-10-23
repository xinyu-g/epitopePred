import pandas as pd
import ast
import glob
import consts as c




def onehot(idx, length):
	lst = [0 for i in range(length)]
	lst[idx] = 1
	return lst 

def zerohot(length):
	return [0 for i in range(length)]

def seq_to_codes(char):
    """Converts sequence to a list of polarity, hydrophobicity, volume"""
    code = list()

    try:
        code = [c.POLARITY[char], c.HYDROPHOBICITY[char], c.VOLUME[char]]
    except KeyError:
        code = [0, 0, 0]

    return code

def get_aa(data, vocab_lst=c.AA_LIST, X='Antigen'):
    length = len(data)
    # standard_data = []
    sequences, labels, masks, features = list(), list(), list(), list()
    for i in range(length):
            antigen = data[X][i]
            Y = data['Y'][i]
            feature = [seq_to_codes(a) for a in antigen]
            sequence = [onehot(vocab_lst.index(s), len(vocab_lst)) for s in antigen]
            label = [0 for i in range(len(antigen))]
            mask = [True for i in range(len(label))]
            # sequence += (maxlength-len(sequence)) * [zerohot(len(vocab_lst))]
            # labels += (maxlength-len(labels)) * [0]
            # mask += (maxlength-len(mask)) * [False]
            for y in Y:
                    label[y] = 1
            features.append(feature)
            sequences.append(sequence)
            labels.append(label)
            masks.append(mask)
            # sequence, labels, mask = sequence[:maxlength], labels[:maxlength], mask[:maxlength]
    return features, sequences, labels, masks
            # sequence, labels, mask = torch.FloatTensor(sequence), torch.FloatTensor(labels), torch.BoolTensor(mask)
            # print(sequence.shape, labels.shape, mask.shape)
            # standard_data.append((sequence, labels, mask))

def get_SS(data, q_lst=c.Q8, S='q8', P='q8_prob'):
    length = len(data)
    SS = list()
    for i in range(length):
        q = list(data[S][i])
        q_prob = [max(prob) for prob in data[P][i]]
        ss = [onehot(q_lst.index(s), len(q_lst)) for s in q]

        ssp = [ss[j] + [q_prob[j]] for j in range(len(ss))]
        SS.append(ssp)
    return SS

def get_features(data, rsa=True, phi=True, psi=True):
    length = len(data)
    features = list()
    for i in range(length):
        RSA, PHI, PSI = list(), list(), list()
        RSA = data['rsa'][i]
        PHI = data['phi'][i]
        PSI = data['psi'][i]
        feature = [[] for i in range(len(RSA))]
        for j in range(len(RSA)):
            if rsa:
                feature[j] += [RSA[j]]
            if phi:
                feature[j] += [PHI[j]]
            if psi:
                feature[j] += [PSI[j]]

        features.append(feature)

    return features
            




