"""Creates and tests model."""


# %% Imports
# %%% Py3 Standard
from pathlib import Path

# %%% 3rd Party
import torch
from torch.utils.data import DataLoader

# %%% User Defined
from mylib import etl, model, train, evaluate, dataset


# %% Variables
# Defaults
EMBED_DIM: int = 64
NUM_EPOCHS: int = 5
BETA: float = 1.0
LEARNING_RATE: float = 2e-4
BATCH_SIZE: int = 50
SCORE_THRESHOLD: float = 0.5
ALPHA: float = 0.5
NUM_WORKERS: int = 2
MODEL_OUTPUT: Path = Path(__file__).parent / 'model_output'


# %% Functions
def main():
    # load data
    # drug_train, sym_train, data_eval, sym_map, pro_map, med_map, ddi, data_test = etl.load_data()
    # symptom_sets, drug_sets_multihot = etl.get_train_data(len(med_map.idx2word))
    # similar_sets = etl.get_similar_set(sym_train)
    # NOTE: tried to HW4-ify this
    train_dataset = dataset.SymptomSetDrugSet('train')
    valid_dataset = dataset.SymptomSetDrugSet('eval')
    test_dataset = dataset.SymptomSetDrugSet('test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.visit_collate_fn, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.visit_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.visit_collate_fn, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model
    # TODO: Still need to define model
    # args here are same as original code.
    mymodel = model.Model(
        train_dataset.n_symptoms,
        train_dataset.n_drugs,
        train_dataset.ddi,
        train_dataset.symptom_set,
        train_dataset.drug_set,
        EMBED_DIM,
    ).to(device)

    # TODO: Repurpose code from HW4 to this use case.
    #  Should give us a tangible way to not plagiarize and still have direction.
    # TODO: Need to define train function
    # TODO: Need to define evaluate function

    # epoch loop
    # see train_seizure and seizure_utils.  these were pulled from HW4 with no modification - yet
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = model.RAdam(mymodel.parameters(), lr=LEARNING_RATE)

    best_val_acc: float = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    for epoch in range(NUM_EPOCHS):
        # train_loss, train_accuracy = train.my_train(model, device, optimizer, sym_train, drug_train, similar_sets, data_eval, len(med_map))
        # valid_loss, valid_accuracy, valid_results = evaluate.my_evaluate(model, data_eval, len(med_map), device)
        # NOTE: more HW4-ing
        train_loss, train_accuracy = train.hw4_train(mymodel, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, _ = evaluate.hw4_evaluate(mymodel, device, valid_loader, criterion)  # should we return valid_results?

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        is_best = valid_accuracy > best_val_acc
        if is_best:
            best_val_acc = valid_accuracy
            torch.save(mymodel, MODEL_OUTPUT, _use_new_zipfile_serialization=False)



    # # evaluate
    evaluate.hw4_evaluate(mymodel, device, test_loader, criterion)

    # plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
    #
    # best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
    # test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)
    #
    # class_names = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen']
    # plot_confusion_matrix(test_results, class_names)


# %% Script
if __name__ == '__main__':
    main()
