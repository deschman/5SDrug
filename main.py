from mylib import etl, model


def main():
    # load train data
    # TODO: Probably will need more data than just this
    # Could we use DataLoader to create batches?  Original implementation is involved.
    drug_train, sym_train = etl.load_train_data()

    # instantiate model
    # TODO: Need to define model
    mymodel = model.Model()

    # TODO: Repurpose code from HW4 to this use case.
    #  Should give us a tangible way to not plagiarize and still have direction.
    # TODO: Need to define train function
    # TODO: Need to define evaluate function

    # epoch loop
    # see train_seizure and seizure_utils.  these were pulled from HW4 with no modification - yet

    # for epoch in range(NUM_EPOCHS):
    # 	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
    # 	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)
    #
    # 	train_losses.append(train_loss)
    # 	valid_losses.append(valid_loss)
    #
    # 	train_accuracies.append(train_accuracy)
    # 	valid_accuracies.append(valid_accuracy)
    #
    # 	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
    # 	if is_best:
    # 		best_val_acc = valid_accuracy
    # 		torch.save(model, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)
    #
    #

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
