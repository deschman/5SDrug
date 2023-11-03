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
We could also use their provided data - this seems like the best use of time.

### Model
Section 3 of the paper details the requirements of the model.  We can also use the aforementioned code base as a guide.

### Evaluation
