# import bepipred2 as bp2
from netsurfp2.__main__ import preprocess, model, parse_fasta
import pandas as pd 
from tdc.single_pred import Epitope, Paratope
import numpy as np
from collections import defaultdict
import consts as c
import time
import logging
import argparse
import glob


log = logging.getLogger('epip')
log.addHandler(logging.NullHandler())

logfmt = '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'    
logging.basicConfig(level=logging.DEBUG,
    format=logfmt, datefmt='%Y-%m-%d %H:%M')


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


def get_feature(protlists, hhblits=c.HHBLITS, hhsuite=c.HHSUITE):
    out_dir = c.OUT_DIR
    searcher = preprocess.HHblits(hhblits, n_threads=20)
    # sec_features = pd.DataFrame()

    log.info('Running {:,} sequences..'.format(len(protlists)))
    computation_start = time.time()
    search_start = time.time()
    
    profiles = searcher(protlists, out_dir)
    search_elapsed = time.time() - search_start
    log.info('Finished profiles ({:.1f} s, {:.1f} s per sequence)..'.format(search_elapsed, search_elapsed / len(protlists)))
    
    pred_start = time.time()
    nsp_model = model.TfGraphModel.load_graph(hhsuite)
    results = nsp_model.predict(profiles, out_dir, batch_size=25)
    pred_elapsed = time.time() - pred_start
    log.info('Finished predictions ({:.1f} s, {:.1f} s per sequence)..'.format(pred_elapsed, pred_elapsed / len(protlists)))
    
    time_elapsed = time.time() - computation_start

    log.info('Total time elapsed: {:.1f} s '.format(time_elapsed))
    log.info('Time per structure: {:.1f} s '.format(time_elapsed / len(protlists)))


    sec_features = pd.DataFrame.from_records(results, index='id')
    sec_features.index.names = ['Antigen_ID']
    sec_features.seq.names = ['Antigen']
    
    return sec_features


def join(data, features):
    log.info('num of records in data: {}, num of records in features: {}'.format(data.shape[0], features.shape[0]))
    drop_columns = ['method', 'desc']
    features.drop(columns=drop_columns, inplace=True)
    res = pd.merge(data, features, how="inner", on=["Antigen_ID"])
    log.info('num of records after join: {}'.format(res.shape[0]))
    
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=c.DATA_RAW + '/*.csv')
    args = parser.parse_args()

    files = glob.glob(args.data)

    for file in files:
        log.info("runing file {}".format(file))
        name = file.split('/')[-1]
        data = pd.read_csv(file)
        protlists = to_protlists(data)
        sec_features = get_feature(protlists)
        joined = join(data, sec_features)
        log.info("writing file: {}".format(name))
        joined.to_csv(c.DATA_SEC + f'/{name}', index='Antigen_ID')
        


if __name__ == "__main__":
    main()


        



    
