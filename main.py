"""Creates and tests model."""


# %% Imports
# %%% Py3 Standard
import time
from pathlib import Path

# %%% 3rd Party
import torch
from torch.utils.data import DataLoader

# %%% User Defined
from mylib import model, train, evaluate, dataset


# %% Variables
# Defaults
EMBED_DIM: int = 64
NUM_EPOCHS: int = 2  # 5
BETA: float = 1.0
LEARNING_RATE: float = 2e-4
BATCH_SIZE: int = 50
SCORE_THRESHOLD: float = 0.5
ALPHA: float = 0.5
NUM_WORKERS: int = 0
MODEL_OUTPUT: Path = Path(__file__).parent / 'model_output'


# %% Functions
def main():
    # load data
    train_dataset: dataset.SymptomSetDrugSet = dataset.SymptomSetDrugSet('train')
    valid_dataset: dataset.SymptomSetDrugSet = dataset.SymptomSetDrugSet('eval')
    test_dataset: dataset.SymptomSetDrugSet = dataset.SymptomSetDrugSet('test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.visit_collate_fn, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.visit_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.visit_collate_fn, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model
    # args here are same as original code.
    mymodel = model.Model(
        train_dataset.n_symptoms,
        train_dataset.n_drugs,
        train_dataset.ddi,
        train_dataset.symptom_set,
        train_dataset.drug_set,
        EMBED_DIM,
    ).to(device)

    # epoch loop
    criterion: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
    optimizer: model.RAdam = model.RAdam(mymodel.parameters(), lr=LEARNING_RATE)  # what is mymodel.parameters()?

    best_val_acc: float = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    for epoch in range(NUM_EPOCHS):
        start_time = time.perf_counter()
        train_loss, train_accuracy = train.hw4_train(mymodel, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, _ = evaluate.hw4_evaluate(mymodel, device, valid_loader, criterion)  # should we return valid_results?

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
    evaluate.hw4_evaluate(mymodel, device, test_loader, criterion)


# %% Script
if __name__ == '__main__':
    main()
