from mylib import etl, model, train, evaluate
import torch

NUM_EPOCHS = 5
EMBED_DIM = 64

def main():
    # load data
    drug_train, sym_train, data_eval, sym_map, pro_map, med_map, ddi = etl.load_data()
    symptom_sets, drug_sets_multihot = etl.get_train_data(len(med_map))
    similar_sets = etl.get_similar_set(sym_train)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    #                                            num_workers=NUM_WORKERS)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    #                                            num_workers=NUM_WORKERS)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    #                                           num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model
    # TODO: Need to define model
    # args here are same as original code.
    mymodel = model.Model(
        len(sym_map),
        len(med_map),
        ddi,
        symptom_sets,
        ddi,
        EMBED_DIM
    ).to(device)

    # TODO: Repurpose code from HW4 to this use case.
    #  Should give us a tangible way to not plagiarize and still have direction.
    # TODO: Need to define train function
    # TODO: Need to define evaluate function

    # epoch loop
    # see train_seizure and seizure_utils.  these were pulled from HW4 with no modification - yet
    optimizer = RAdam(model.parameters(), lr=args.lr)

    for epoch in range(NUM_EPOCHS):
    	train_loss, train_accuracy = train.my_train(model, device, optimizer, sym_train, drug_train, similar_sets)
    	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

    	train_losses.append(train_loss)
    	valid_losses.append(valid_loss)

    	train_accuracies.append(train_accuracy)
    	valid_accuracies.append(valid_accuracy)

    	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
    	if is_best:
    		best_val_acc = valid_accuracy
    		torch.save(model, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)



    # # evaluate

    # plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
    #
    # best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
    # test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)
    #
    # class_names = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen']
    # plot_confusion_matrix(test_results, class_names)


if __name__ == '__main__':
    main()
