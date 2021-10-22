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

def get_aa(data, vocab_lst, X='Antigen'):
    length = len(data)
    # standard_data = []
    for i in range(length):
            antigen = data[X][i]
            Y = data['Y'][i]

            sequence = [onehot(vocab_lst.index(s), len(vocab_lst)) for s in antigen]
            labels = [0 for i in range(len(antigen))]
            mask = [True for i in range(len(labels))]
            # sequence += (maxlength-len(sequence)) * [zerohot(len(vocab_lst))]
            # labels += (maxlength-len(labels)) * [0]
            # mask += (maxlength-len(mask)) * [False]
            for y in Y:
                    labels[y] = 1
            # sequence, labels, mask = sequence[:maxlength], labels[:maxlength], mask[:maxlength]
    return sequence, labels, mask
            # sequence, labels, mask = torch.FloatTensor(sequence), torch.FloatTensor(labels), torch.BoolTensor(mask)
            # print(sequence.shape, labels.shape, mask.shape)
            # standard_data.append((sequence, labels, mask))

def get_q8(data, X='q8'):
    pass

def get_q3(data):
    pass

def get_rsa(data):
    pass

def get_phi(data):
    pass

def get_psi(data):
    pass

def seq_to_codes(char):
    """Converts sequence to a list of polarity, hydrophobicity, volume"""
    code = list()

    try:
        code = [c.POLARITY[char], c.HYDROPHOBICITY[char], c.VOLUME[char]]
    except KeyError:
        code = [0, 0, 0]

    return code


