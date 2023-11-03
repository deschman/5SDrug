import dill
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

def load_train_data():
    # TODO: Make tensor?
    drug_train = dill.load(open('data/drug_train.pkl', 'rb'))
    sym_train = dill.load(open('data/sym_train.pkl', 'rb'))
    # x = dill.load(open('data/ddi_A_final.pkl', 'rb'))
    return drug_train, sym_train