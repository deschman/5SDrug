"""Defines SymptomSetDrugSet class and associated helper functions"""


# %% Imports
# %%% Py3 Standard
from pathlib import Path
from typing import Tuple, Any, Dict, List, Literal

# %%% 3rd Party
import numpy as np
import dill
import torch
from torch.utils.data import Dataset
from scipy import sparse


# %% Classes
class SymptomSetDrugSet(Dataset):
    """
    Loads the symptom and drug sets for training, evaluation, and testing.

    Params
    ------
    dataset : Literal['train', 'eval', 'test']
        The type of dataset to load.

    Properties
    ----------
    n_symptoms : int
        The number of symptoms in the dataset
    n_drugs : int
        The number of drugs in the dataset
    symptom_set : List[sparse.coo_array]
        The symptom set for each patient.
    drug_set : List[sparse.coo_array]
        The drug set for each patient.
    """
    def __init__(self, dataset: Literal['train', 'eval', 'test']) -> None:

        data_path: Path = Path(__file__).parent.parent / 'data'

        self.ddi = dill.load(
            open(
                data_path / 'ddi_A_final.pkl',
                'rb',
            )
        )

        self.voc: Dict[str, object] = dill.load(
            open(
                data_path / 'voc_final.pkl',
                'rb',
            )
        )
        self.n_symptoms: int = len(self.voc['sym_voc'])
        self.n_drugs: int = len(self.voc['diag_voc'])

        all_training_data: List[List[int]] = dill.load(
            open(
                data_path / f'data_{dataset}.pkl',
                'rb',
            )
        )
        self.symptom_set: List[sparse.coo_array] = [
            sparse.coo_array(
                (
                    [1 for _ in all_training_data[p][0]],
                    all_training_data[p][0],
                ),
                self.n_symptoms,
            )
            for p in range(len(all_training_data))
        ]
        self.drug_set: List[sparse.coo_array] = [
            sparse.coo_array(
                (
                    [1 for _ in all_training_data[p][1]],
                    all_training_data[p][1],
                ),
                self.n_symptoms,
            )
            for p in range(len(all_training_data))
        ]

    def __len__(self) -> int:
        return len(self.symptom_set)

    def __getitem__(self, index: int) -> Tuple[Any]:
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.symptom_set[index], self.drug_set[index]

def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
    where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

    Returns
    -------
    seqs : FloatTensor
        3D of batch_size X max_length X num_features
    lengths : LongTensor
        1D of batch_size
    labels : LongTensor
        1D of batch_size
    """
    max_rows: int = max([b[0].shape[1] for b in batch])

    lengths = [b[0].shape[1] for b in batch]
    seqs = [
        np.pad(batch[b][0].toarray(), ((0, max_rows - batch[b][0].shape[1]), (0, 0))) if len(batch) >= b + 1
        else np.zeros((batch[0][0].shape[1], max_rows))
        for b in range(len(batch))
    ]
    labels = [b[1] for i, b in enumerate(batch) if i < len(batch)]

    seqs_tensor = torch.LongTensor(seqs)
    lengths_tensor = torch.LongTensor(lengths)
    labels_tensor = torch.LongTensor(labels)

    return (seqs_tensor, lengths_tensor), labels_tensor
