import dill
import numpy as np
import os
'''
#### TODO, if we don't use their data
* We need to extract symptoms from the clinical texts per 5.1.1.  The original uses a NLP technique 
from [this cited paper](https://www.researchgate.net/publication/338552712_PIC_a_paediatric-specific_intensive_care_database).  We likely will need to use a poor man's approach to this.

* We will need DDI info per 3.4.2 and 5.1.1.  It seems we can get the DDI info via 
`from tdc.multi_pred import DDI
data = DDI(name = 'TWOSIDES')
split = data.get_split()`

* We will need to ingest the relevant data into tensors.  We can probably reuse code from hw4 for this.

* We will need to split our data 4x1x1 into training, validation, and testing sets. 

'''

def load_data():
    # TODO: Make tensor?

    drug_train = dill.load(open('data/drug_train_50.pkl', 'rb'))
    sym_train = dill.load(open('data/sym_train_50.pkl', 'rb'))
    data_eval = dill.load(open('data/data_eval.pkl', 'rb'))
    # these voc objects have 2 properties - idx2word and word2idx.  Basically a feature mapping
    voc = dill.load(open('data/voc_final.pkl', 'rb'))
    sym_map, pro_map, med_map = voc['sym_voc'], voc['diag_voc'], voc['med_voc']

    ddi = dill.load(open('data/ddi_A_final.pkl', 'rb'))
    return drug_train, sym_train, data_eval, sym_map, pro_map, med_map, ddi

def get_train_data(n_drug):
    data_train = dill.load(open('data/drug_train.pkl', 'rb'))
    symptom_sets, drug_sets_multihot = [], []
    for adm in data_train:
        syms, drugs = adm[0], adm[2]
        symptom_sets.append(syms)
        drug_multihot = np.zeros(n_drug)
        drug_multihot[drugs] = 1
        drug_sets_multihot.append(drug_multihot)
    return

def get_similar_set(sym_train):
    similar_sets = [[] for _ in range(len(sym_train))]
    for i in range(len(sym_train)):
        for j in range(len(sym_train[i])):
            similar_sets[i].append(j)

    for idx, sym_batch in enumerate(sym_train):
        if len(sym_batch) <= 2 or len(sym_batch[0]) <= 2: continue
        batch_sets = [set(sym_set) for sym_set in sym_batch]
        for i in range(len(batch_sets)):
            max_intersection = 0
            for j in range(len(batch_sets)):
                if i == j: continue
                if len(batch_sets[i] & batch_sets[j]) > max_intersection:
                    max_intersection = len(batch_sets[i] & batch_sets[j])
                    similar_sets[idx][i] = j

    return similar_sets