# 5SDrug
## 5SDrug: Supplemental Symptom-based Set-to-set Small and Safe Drug Recommendation

This repository contains code that attempts to re-create the results of Tan et al in ["4SDrug: Symptom-based Set-to-set Small and Safe Drug Recommendation"](https://dl.acm.org/doi/abs/10.1145/3534678.3539089). The code contained within attempts to achieve the same results as the aformentioned paper with special care taken to avoid plagiarizing the [code base](https://github.com/Melinda315/4SDrug) created for the previous work.

The ["Overleaf project"](https://www.overleaf.com/project/64ed0e08cd636777ff5ceb63) will house all written work.

## Environment
* torch
* dill
* numpy
* scipy

## Implementation
Run the main.py file to run our code.

### Data
Data is sourced from the [MIMIC website](https://physionet.org/content/mimiciii/1.4/). However, we use the preprocessed data found in the original code base linked above. 

### Model
Section 3 of the paper details the requirements of the model.  We use the original implementation.

