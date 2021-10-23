HHBLITS = "/home/xinyugu4/data/Databases/hhsuite/uniclust30_2017_04/uniclust30_2017_04"
HHSUITE = "/home/xinyugu4/netsurfp2/models/hhsuite.pb"
OUT_DIR = "/home/xinyugu4/out/epip_out"
DATA_SEC = "/home/xinyugu4/data/Datasets/epip/sec"
DATA_RAW = "/home/xinyugu4/data/Datasets/epip/raw"
TRAIN_OUT = "/home/xinyugu4/out/epip_out/train"

AA_LIST = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'B', 'Z', 'J']

#Article: Comparing the polarities of the amino acids: Side-chain distribution coefficients between the vapor phase, cyclohexane, 1-octanol, and neutral aqueous solution
POLARITY       = {'A': -0.06, 'R': -0.84, 'N': -0.48, 'D': -0.80, 'C': 1.36, 'Q': -0.73,
                  'E': -0.77, 'G': -0.41, 'H': 0.49, 'I': 1.31, 'L': 1.21, 'K': -1.18, 
                  'M': 1.27, 'F': 1.27, 'P': 0.0, 'S': -0.50, 'T': -0.27, 'W': 0.88, 
                  'Y': 0.33, 'V': 1.0}
#Article: Correlation of sequence hydrophobicities measures similarity in three-dimensional protein structure
HYDROPHOBICITY = {'A': -0.40, 'R': -0.59, 'N': -0.92, 'D': -1.31, 'C': 0.17, 'Q': -0.91,
                  'E': -1.22, 'G': -0.67, 'H': -0.64, 'I': 1.25, 'L': 1.22, 'K': -0.67, 
                  'M': 1.02, 'F': 1.92, 'P': -0.49, 'S': -0.55, 'T': -0.28, 'W': 0.50, 
                  'Y': 1.67, 'V': 0.91}
#Article: On the average hydrophobicity of proteins and the relation between it and protein structure
VOLUME         = {'A': 52.6, 'R': 109.1, 'N': 75.7, 'D': 68.4, 'C': 68.3, 'Q': 89.7,
                  'E': 84.7, 'G': 36.3, 'H': 91.9, 'I': 102.0, 'L': 102.0, 'K': 105.1, 
                  'M': 97.7, 'F': 113.9, 'P': 73.6, 'S': 54.9, 'T': 71.2, 'W': 135.4, 
                  'Y': 116.2, 'V': 85.1, 'X': 52.6, 'Z': 52.6, 'B': 52.6, 'J': 102.0}


Q8 = ['G', 'H', 'I', 'B', 'E', 'S', 'T', 'C']

Q3 = ['H', 'E', 'C']

