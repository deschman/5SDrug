"""Creates and tests model."""


# %% Imports
# %%% Py3 Standard
import time
from pathlib import Path
from typing import List, Dict, Iterable, Any

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
NUM_EPOCHS: int = 10
NUM_PARAMS: int = 10

EMBED_DIM: List[int] = [16 * (i + 1) for i in range(NUM_PARAMS)]
LEARNING_RATE: List[float] = [5e-5 * (i + 1) for i in range(NUM_PARAMS)]
BATCH_SIZE: List[int] = [12 * (i + 1) for i in range(NUM_PARAMS)]
ALPHA: List[float] = [0.12 * (i + 1) for i in range(NUM_PARAMS)]
BETA: List[float] = [0.12 * (i + 1) for i in range(NUM_PARAMS)]


# %% Functions
def main(config: Dict[str, Any]) -> None:
    # load data
    train_dataset: dataset.SymptomSetDrugSet = dataset.SymptomSetDrugSet('train')
    valid_dataset: dataset.SymptomSetDrugSet = dataset.SymptomSetDrugSet('eval')
    test_dataset: dataset.SymptomSetDrugSet = dataset.SymptomSetDrugSet('test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=dataset.visit_collate_fn, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=dataset.visit_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=dataset.visit_collate_fn, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model
    mymodel = model.Model(
        train_dataset.n_symptoms,
        train_dataset.n_drugs,
        train_dataset.ddi,
        config['embed_dim'],
    ).to(device)

    # epoch loop
    optimizer: model.RAdam = model.RAdam(mymodel.parameters(), lr=config['learning_rate'])

    best_val_acc: float = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        train_loss, train_accuracy = train.hw4_train(
            mymodel,
            device,
            train_loader,
            optimizer,
            epoch,
            config['alpha'],
            config['beta'],
        )
        valid_loss, valid_accuracy, _ = evaluate.hw4_evaluate(
            mymodel,
            device,
            valid_loader,
            config['alpha'],
            config['beta'],
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        end_time = time.perf_counter()
        print(end_time - start_time)

        is_best = valid_accuracy > best_val_acc
        if is_best:
            best_val_acc = valid_accuracy
            torch.save(mymodel, MODEL_OUTPUT, _use_new_zipfile_serialization=False)

    # evaluate
    results = evaluate.hw4_evaluate(
        mymodel,
        device,
        test_loader,
        config['alpha'],
        config['beta'],
    )

    return {'f1': results[0], 'loss': results[1], 'accuracy': results[2]}


# %% Script
if __name__ == '__main__':
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
