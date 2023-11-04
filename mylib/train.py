import torch, time, numpy as np
from mylib import evaluate
ALPHA = 0.5
BETA = 1.0
def my_train(model, device, optimizer, sym_train, drug_train, similar_sets_idx):

	model.train()
	losses = 0.0

	# training loop
	for i, (syms, drugs, similar_idx) in enumerate(zip(sym_train, drug_train, similar_sets_idx)):
		model.zero_grad()
		optimizer.zero_grad()
		scores, bpr, loss_ddi = model(syms, drugs, similar_idx, device)

		loss = evaluate.custom_criterion(scores, bpr, loss_ddi, drugs, ALPHA, BETA, device)
		losses += loss.item() / syms.shape[0]
		loss.backward()
		optimizer.step()

		ja, prauc, avg_p, avg_r, avg_f1, avg_med, ddi_rate = evaluate.my_evaluate(
			model, pklSet.data_eval, pklSet.n_drug, device
		)
	return train_loss, train_accuracy


def hw4_train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		#print('output', output.shape, 'target', target.shape)
		loss = criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg