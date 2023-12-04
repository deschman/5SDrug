"""Creates and tests model."""


# %% Imports
# %%% Py3 Standard
import time
from pathlib import Path
from typing import List, Dict, Iterable, Any, Tuple

# %%% 3rd Party
import torch
from torch.utils.data import DataLoader
from ray import tune

# %%% User Defined
from mylib import model, train, evaluate, dataset


# %% Variables
# Defaults
NUM_WORKERS: int = 0
MODEL_OUTPUT: Path = Path(__file__).parent / 'model_output'
NUM_EPOCHS: int = 200

EMBED_DIM: List[int] = [32]  # [32, 24]
LEARNING_RATE: List[float] = [0.1]  # [0.1, 0.01]
BATCH_SIZE: List[int] = [100]  # [500]
ALPHA: List[float] = [0.5]  # [0.5, 0.25]
BETA: List[float] = [0.75]
DEFAULT_CONFIG: Dict[str, Any] = {
    'embed_dim': EMBED_DIM[0],
    'learning_rate': LEARNING_RATE[0],
    'batch_size': BATCH_SIZE[0],
    'alpha': ALPHA[0],
    'beta': BETA[0],
}


# %% Functions
def main(config: Dict[str, Any] = DEFAULT_CONFIG) -> None:
    # load data
    train_dataset: dataset.SymptomSetDrugSet = dataset.SymptomSetDrugSet('train')
    valid_dataset: dataset.SymptomSetDrugSet = dataset.SymptomSetDrugSet('eval')
    test_dataset: dataset.SymptomSetDrugSet = dataset.SymptomSetDrugSet('test')

    train_loader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=dataset.visit_collate_fn,
        num_workers=NUM_WORKERS,
    )
    valid_loader: DataLoader = DataLoader(
        dataset=valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=dataset.visit_collate_fn,
        num_workers=NUM_WORKERS,
    )
    test_loader: DataLoader = DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=dataset.visit_collate_fn,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model
    mymodel: model.Model = model.Model(
        train_dataset.n_symptoms,
        train_dataset.n_drugs,
        train_dataset.ddi,
        config['embed_dim'],
    ).to(device)

    # epoch loop
    optimizer: model.RAdam = model.RAdam(
        mymodel.parameters(),
        lr=config['learning_rate'],
    )

    best_f1: float = 0.0
    for epoch in range(NUM_EPOCHS):
        train.hw4_train(
            mymodel,
            device,
            train_loader,
            optimizer,
            epoch,
            config['alpha'],
            config['beta'],
        )

        if epoch + 1 % 5 == 0:
            start_time: float = time.perf_counter()
            val: Tuple[float, float, float, float, int] = evaluate.hw4_evaluate(
                mymodel,
                device,
                valid_loader,
                config['alpha'],
                config['beta'],
            )

            end_time: float = time.perf_counter()
            print(f'Epoch: [{epoch}/{NUM_EPOCHS}] Time {end_time - start_time}')

            if val[0] > best_f1:
                best_f1 = val[0]
                torch.save(
                    mymodel,
                    MODEL_OUTPUT,
                    _use_new_zipfile_serialization=False,
                )

    # evaluate
    test: Tuple[float, float, float, float, int] = evaluate.hw4_evaluate(
        mymodel,
        device,
        test_loader,
        config['alpha'],
        config['beta'],
    )

    return {
        'f1': test[0],
        'loss': test[1],
        'accuracy': test[2],
        'drug_count': test[3],
        'ddi_count': test[4],
    }


# %% Script
if __name__ == '__main__':
    # main()
    config: Dict[str, Dict[str, Iterable[Any]]] = {
        'embed_dim': tune.grid_search(EMBED_DIM),
        'learning_rate': tune.grid_search(LEARNING_RATE),
        'batch_size': tune.grid_search(BATCH_SIZE),
        'alpha': tune.grid_search(ALPHA),
        'beta': tune.grid_search(BETA),
    }
    tuner: tune.Tuner = tune.Tuner(main, param_space=config)
    results: tune.ResultGrid = tuner.fit()
    print(results.get_best_result(metric="f1", mode="max").config)
