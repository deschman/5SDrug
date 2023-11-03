# 5SDrug
## 5SDrug: Supplemental Symptom-based Set-to-set Small and Safe Drug Recommendation

This repository contains code that attempts to re-create the results of Tan et al in ["4SDrug: Symptom-based Set-to-set Small and Safe Drug Recommendation"](https://dl.acm.org/doi/abs/10.1145/3534678.3539089). The code contained within attempts to achieve the same results as the aformentioned paper with special care taken to avoid plagiarizing the [code base](https://github.com/Melinda315/4SDrug) created for the previous work.

The ["Overleaf project"](https://www.overleaf.com/project/64ed0e08cd636777ff5ceb63) will house all written work.

## Environment
Use `conda env create -f environment.yml` to set up the environment from HW4.  The implementation of the paper uses PyTorch so this should be a good start.  We can edit the environment.yml as we make changes.

Updating the env seems to be possible with `conda env update --name proj --file environment.yml --prune` 
per [this SO](https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file).
We could also change the name in environment.yml everyime it is changed (for example, proj1, proj2, ...) and just make sure we stay in sync.

## Implementation
This is a multi-label binary classification task. It looks like HW4 will be useful.

### Data ETL
Data is sourced from the [MIMIC website](https://physionet.org/content/mimiciii/1.4/). Documentation can be found there as well.  The provided code base only provides the datasets, so we have no guide for this.  
We could also punt and callout that we used their curated datasets from the code base if we wanted to dedicate our time to the model.
#### TODO
* Punt?

* We need to extract symptoms from the clinical texts per 5.1.1.  The original uses a NLP technique 
from [this cited paper](https://www.researchgate.net/publication/338552712_PIC_a_paediatric-specific_intensive_care_database).  We likely will need to use a poor man's approach to this.

* We will need DDI info per 3.4.2 and 5.1.1.  It seems we can get the DDI info via 
`from tdc.multi_pred import DDI
data = DDI(name = 'TWOSIDES')
split = data.get_split()`

* We will need to ingest the relevant data into tensors.  We can probably reuse code from hw4 for this.

* We will need to split our data 4x1x1 into training, validation, and testing sets. 

### Model
Section 3 of the paper details the requirements of the model.  We can also use the aforementioned code base as a guide.

#### TODO
* Average pooling will be used to represent sets of drugs and / or symptoms.
### Evaluation
