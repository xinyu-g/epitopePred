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




data =  Epitope(name = 'IEDB_Jespersen')
split = data.get_split()
train_data = split['train']
valid_data = split['valid']
test_data = split['test']


class Alphafold:

    def __init__(self, db, script, n_threads=None):
        self.db = db
        self.script = script
        self.n_threads = int(n_threads) if n_threads else None

    def call(self, protlist, output_dir):
        profiles = {}
        for protid, seq in protlist:
            self.search()

    def search(self, protid, seq, output_dir):
        fa_name = os.path.join(output_dir, protid)

        with open(fa_name, 'w') as fa_file:
            print('>' + protid, file=fa_file)
            print(seq, file=fa_file)


        cmd = [
            'bash', self.script, '-d', self.db, '-o', output_dir, '-f', fa_name 
        ]




