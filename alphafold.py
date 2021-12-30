from tdc.single_pred import Epitope
import tempfile
import multiprocessing
import os
import pickle
import sqlite3
import logging
import subprocess
import collections
import numpy as np
import consts as c
import argparse
from collections import defaultdict
import time
from pathlib import Path


log = logging.getLogger('alphafold')
log.addHandler(logging.NullHandler())

logfmt = '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'   





class Alphafold:

    def __init__(self, db, script, n_threads=None):
        self.db = db
        self.script = script
        self.n_threads = int(n_threads) if n_threads else None

    # def call_feature(self, protlist, output_dir):
    #     profiles = {}
    #     for protid, seq in protlist:
    #         self.feature()



    def feature(self, protid, seq, output_dir):
        fa_name = os.path.join(output_dir, protid + '.fasta')
        log_name = os.path.join(output_dir, protid + '_feature.log')

        with open(fa_name, 'w') as fa_file:
            print('>' + protid, file=fa_file)
            print(seq, file=fa_file)


        cmd = [
            'bash', self.script, '-d', self.db, '-o', output_dir, '-p', 'monomer_ptm', '-i', fa_name, '-m', 'model_1', '-f'
        ]

        with open(log_name, 'w') as logfp:
            skw = dict(stdout=logfp, stderr=logfp)
            p = subprocess.run(cmd, universal_newlines=True, **skw)


        p.check_returncode()


    def model(self, protid, seq, output_dir):
        fa_name = os.path.join(output_dir, protid + '.fasta')
        log_name = os.path.join(output_dir, protid + '_model.log')

        cmd = [
            'bash', self.script, '-d', self.db, '-o', output_dir, '-p', 'monomer_ptm', '-i', fa_name
        ]


        with open(log_name, 'w') as logfp:
            skw = dict(stdout=logfp, stderr=logfp)
            p = subprocess.run(cmd, universal_newlines=True, **skw)


        p.check_returncode()

def to_protlists(data, type='dict'):

    protlists = list()
    if type == 'dict':
        protlists = {}
    temp_dict = data.to_dict("records")
    for record in temp_dict:
        protlist = defaultdict(list)
        antigen = record['Antigen_ID']
        seq = record['Antigen']
        desc = ''
        protlist[antigen].append(desc)
        protlist[antigen].append(seq)

        if type == 'list':
            protlists.append(protlist)
        elif type == 'dict':
            protlists.update(protlist)
        
    return protlists

def call(protlists, db, script, out, step):
    alpha_model = Alphafold(db, script)
    for protid, (desc, seq) in protlists.items():
        protid = protid.replace(" ", "_")
        if step == 'feature':
            log.info('Featuring {}'.format(protid))
            feature_start = time.time()
            alpha_model.feature(protid, seq, out)
            feature_elapsed = time.time() - feature_start
            log.info('Finished Feature {:.1f} s'.format(feature_elapsed))
        else:
            m_path = Path(out + '/' + protid + '/ranked_4.pdb')
            if m_path.is_file():
                continue 
            log.info('Modeling {}'.format(protid))
            model_start = time.time() 
            alpha_model.model(protid, seq, out)
            model_elapsed = time.time() - model_start
            log.info('Finished model {:.1f} s'.format(model_elapsed))

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=c.ALPHA_DATA)
    # parser.add_argument('--out', default=c.ALPHA_OUT)
    parser.add_argument('--step', default='feature')
    parser.add_argument('--script', default=c.ALPHA_SCRIPT)
    args = parser.parse_args()

    logging.basicConfig(
    level=logging.DEBUG,
    format=logfmt, 
    datefmt='%Y-%m-%d %H:%M',
    handlers=[
        logging.FileHandler(filename=f'{c.ALPHA_OUT}/{args.step}.log', mode='a'), # mode='w+' overwrite
        logging.StreamHandler()
    ])

    data =  Epitope(name = 'IEDB_Jespersen')
    split = data.get_split()
    train_data = split['train']
    valid_data = split['valid']
    test_data = split['test']

    train_protlists = to_protlists(train_data)
    valid_protlists = to_protlists(valid_data)
    test_protlists =  to_protlists(test_data)
    # log.info("ptotlist {}".format(test_protlists))
    call(train_protlists, args.data, args.script, c.ALPHA_OUT + 'train', args.step)
    call(valid_protlists, args.data, args.script, c.ALPHA_OUT + 'valid', args.step)
    call(test_protlists, args.data, args.script, c.ALPHA_OUT + 'test', args.step)


if __name__ == '__main__':
    main()

    


    