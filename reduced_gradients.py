import torch
from torch import linalg, nn
from torch.nn import functional as f
from config import *
from lyapunov import *
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors, cm
import imageio
from torch.distributions import Categorical
import os
import math
from training import *
from lyapunov import param_split
from collections import OrderedDict



if __name__ == '__main__':
	print('Starting')
	batch_size = 128
	le_batch_size = 25
	output_size = 10
	max_epoch = 4
	learning_rate = 0.002
	dropout = 0.1
	hidden_size = 16
	save_interval = 1
	model_type = 'gru'
	p = 0.001
	seq_length = 112
	input_size = 7
	max_sub_epoch = 1
	in_epoch_saves = 4
	lr = 0.02
	overwrite = True
	overwrite_losses = True
	calc_vals = True
	
	# params = torch.linspace(0.005, end= 0.025, steps = 2)
	params = [0.001, 0.005, 0.025, 0.05, 0.25, 0.5]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
	dcon = DataConfig('../Dataset/', input_size = input_size, batch_size= batch_size, input_seq_length = seq_length, 
												target_seq_length = 1, val_frac = 0.2, 
												test_frac = 0, name_params = {'insize':input_size})
	mcon = ModelConfig(model_type, 1, hidden_size, dcon.input_size, output_size = output_size, dropout=dropout, 
						init_type = 'normal', init_params = {'mean':0, 'std':p},
						device = device, bias = False, id_init_param = 'std')                                            
	tcon = TrainConfig(model_dir = 'SMNIST/Models', batch_size = batch_size, max_epoch = max_epoch, 
														optimizer = 'sgd', learning_rate = learning_rate)
	fcon = FullConfig(dcon, tcon, mcon)
	
	
	if in_epoch_saves >0:
		if os.path.exists('SMNIST/training_saveIdcs.p'):
			save_idcs = torch.load('SMNIST/training_saveIdcs.p')
		else:
			train_dataloader = torch.utils.data.DataLoader(fcon.data.datasets['train_set'], 
															batch_size = fcon.train.batch_size)
			epoch_samples = len(list(train_dataloader))
			save_idcs = part_equal(epoch_samples, in_epoch_saves)
			torch.save(save_idcs, 'SMNIST/training_saveIdcs.p')
	else:
		save_idcs = []
		
	val_dl = torch.utils.data.DataLoader(fcon.data.datasets['val_set'], 
													batch_size = le_batch_size)
	le_input, le_target = torch.load('SMNIST/le_setup.p', map_location = device)
	# print(f'Input shape: {le_input.shape}')
	le_input = le_input.to(fcon.device).squeeze(1)
	le_target = le_target.to(fcon.device)
	print(le_input.shape)
	
	criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
	for p in params:
		plots_folder = f'SMNIST/Plots/p{p}' 
		if not os.path.exists(plots_folder):
			os.makedirs(plots_folder)
		p = float(int(p*1000))*1.0/1000
		fcon.model.init_params['std'] = p
		print(f'Parameter = {p}')
		for epoch in range(1, max_epoch+1, save_interval):
			print(f"Epoch {epoch}")
			if calc_vals:
				for it in range(in_epoch_saves + 1):
					if it == in_epoch_saves:
						model, optimizer, train_loss, _ = load_checkpoint(fcon, epoch)
						suffix = ''
						it_lab = ''
					elif epoch == 0 or epoch >= max_sub_epoch:
						continue
					else:
						ind = save_idcs[it]
						print(f'Iteration: {ind}')
						model, optimizer, train_loss = load_iter_checkpoint(fcon, epoch, save_idcs[it])
						suffix = f'_iter{ind}'
						it_lab = f', Iteration {ind}'
					ftle_dict = torch.load(f'SMNIST/LEs/{fcon.name()}_e{epoch}{suffix}_FTLE.p', map_location = device)
					gradV_list, gradU_list, gradW_list, loss_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_Fullgrads.p', map_location = device)
					gradV_list_new, gradU_list_new, gradW_list_new = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_Newgrads.p', map_location = device)
					pred_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_logits.p', map_location = device)
					h_list = ftle_dict['h_list']
					x_list = ftle_dict['x_list']
					ftles = ftle_dict['FTLE']
					print(f'ftle shape: {ftles.shape}')
					h_le = ftle_dict['h'].to(fcon.device)
					# print(f'Input diff: {torch.norm(x_list - le_input)}')
					new_loss_name = f'SMNIST/Grads/p{p}_newLoss_e{epoch}{suffix}.p'
					new_loss = torch.load(new_loss_name)
					
					outputs, h_t = model(le_input, h_le)
					init_loss = criterion(outputs, le_target)
					# print(f'Full Loss: {init_loss}')
					# print(f'Init Loss Shape: {init_loss.shape}')
					# print(f'Grad V Diff: {torch.linalg.norm(gradV_list_new - gradV_list)}')
					# print(f'Grad W Diff: {torch.linalg.norm(gradW_list_new - gradW_list)}')
					# print(f'Grad U Diff: {torch.linalg.norm(gradU_list_new - gradU_list)}')
					
					fname = f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_ReducedLosses.p'
					ranks = torch.arange(hidden_size+1)
					state_dict = model.rnn_layer.state_dict()
					V = state_dict['weight_hh_l0']
					U = state_dict['weight_ih_l0']
					W = model.fc.weight
					# print(f'Init V: {V}')
					all_norms_list = []
					ftle_means_list = []
					ftle_max_list = []
		
					if os.path.exists(fname) and overwrite == False:
						dV_losses = torch.load(fname).to(device)
					else:
						
						dV_losses = torch.zeros(len(ranks), le_batch_size).to(device)
						
						#Calculate loss reduction from rank-r delta-V's
						
						model.eval()
						# print(model.rnn_layer.all_weights)
						with torch.no_grad():
							init_V = V.data.clone().detach()
							init_W = W.data.clone().detach()
							init_U = U.data.clone().detach()
							gradQ_norms = []
							for q in tqdm(ranks):
								red_loss_name = f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_ReducedLosses.p'
								if q == -1:
									gradQ = torch.load(f'SMNIST/Grads/p{p}_h{hidden_size}_gradVQRFull.p')
									# print(f'Grad Diff: \n {(gradV_list[0].cpu()-gradQ[0])}')
								else:
									gradQ = torch.load(f'SMNIST/Grads/p{p}_h{hidden_size}_gradVQR_red{q}.p')
									# print(f'gradQ: {gradQ[0]}')
								gradQ_norms.append(torch.linalg.norm(gradQ, dim = (-2, -1)))
								if q >0:
									ftle_means_list.append(ftles[:, -1, :q].mean(dim = -1))
									ftle_max_list.append(ftles[:,-1,:q].max(dim = -1)[0])
								else:
									ftle_means_list.append(torch.zeros_like(ftles[:, -1, 0]).squeeze())
									ftle_max_list.append(torch.zeros_like(ftles[:, -1, 0]).squeeze())
								# print(f'Norms shape: {gradQ_norms[-1].shape}')
								if os.path.exists(red_loss_name) and not overwrite_losses:
									continue
								for batch in range(le_batch_size):
									# print(f'V: {V}')
									V_b = init_V - lr*gradQ[batch].to(device)
									W_b = init_W - lr*gradW_list[batch].to(device).transpose(-2,-1)
									U_b = init_U - lr*gradU_list[batch].to(device)
									new_state = OrderedDict([(k, V_b) if k == 'weight_hh_l0' else (k, v) for k, v in state_dict.items()])
									new_state = OrderedDict([(k, U_b) if k == 'weight_ih_l0' else (k, v) for k, v in new_state.items()])
									model.rnn_layer.load_state_dict(new_state, strict = False)
									model.fc.weight = torch.nn.parameter.Parameter(W_b)
									newW = model.fc.weight
									if batch == le_batch_size-1 and q == 0:
										W_diff = newW - W
										# print(f'W Difference: {(W_diff/newW).norm()}')
									preds, _ = model(x_list[batch].unsqueeze(0), h_le[:, batch].unsqueeze(1))
									temp_loss = criterion(preds, le_target[batch].unsqueeze(0))
									dV_losses[q, batch] = temp_loss
							torch.save(dV_losses, f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_ReducedLosses.p')
							all_norms_list.append(torch.stack(gradQ_norms))
					ftle_means_list = torch.stack(ftle_means_list).squeeze()
					ftle_max_list = torch.stack(ftle_max_list).squeeze()
					all_norms_list = torch.stack(all_norms_list).squeeze()
					full_loss_diffs = (dV_losses-dV_losses[-1].unsqueeze(0).repeat(hidden_size+1, 1)).detach()
					torch.save(all_norms_list, f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_AllNorms.p')
					torch.save(full_loss_diffs, f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_LossDiffs.p')
			
			# Plot: gradV dimension vs. Loss difference (across samples) as a function of rank
			plt.figure(figsize = (8,4))
			num_plot = 25
			all_norms_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_AllNorms.p')
			full_loss_diffs = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_LossDiffs.p')
			for i in range(num_plot):
				plt.plot(torch.arange(hidden_size+1), (dV_losses.detach().cpu()-init_loss.detach().cpu())[:, i])
			plt.xlabel('gradV Dimension')
			plt.ylabel('Loss Difference')
			plt.title(f'Losses by dimension for p = {p}, epoch = {epoch}')
			plt.savefig(f'{plots_folder}/p{p}_QLosses.png')
			plt.close()
			
			# Plot: Loss Comparison across gradient types
			plt.figure(figsize = (8,4))
			plt.xlabel('Sample No.')
			plt.scatter(torch.arange(le_batch_size), dV_losses[-1].detach().cpu(), label = 'dV Loss')
			plt.scatter(torch.arange(le_batch_size), new_loss.detach().cpu(), label = 'PT Loss')
			plt.scatter(torch.arange(le_batch_size), init_loss.detach().cpu(), label = 'Init Loss')
			plt.legend()
			plt.ylabel('Loss')
			plt.title(f'Updated Losses for p = {p}')
			plt.savefig(f'{plots_folder}/p{p}_e{epoch}_UpdatedLosses.png', dpi = 400, bbox_inches = 'tight')
			plt.close()
			
			#Plot: Cosine similarity vs Grad Rank ()
			q_list = torch.arange(hidden_size+1)
			cosines = torch.load(f'SMNIST/Grads/p{p}_e{epoch}{suffix}_cosines.p')
			nc  = colors.Normalize()
			nc.autoscale(full_loss_diffs.cpu())
			plt.figure(figsize = (8,4))
			plt.scatter(q_list.unsqueeze(1).repeat(1,le_batch_size), cosines.detach().cpu(), c= full_loss_diffs.cpu(), norm = nc)
			plt.title(f'Cosine similarity with Full Gradient, p = {p}, epoch = {epoch}')
			plt.ylabel('Cosine Similarity')
			plt.xlabel('Grad Rank')
			plt.colorbar()
			plt.savefig(f'{plots_folder}/p{p}_e{epoch}_CosineSimilarity.png', dpi = 400, bbox_inches = 'tight')
			plt.close()
			
			plt.figure(figsize = (8,4))
			# print(f'Cosines shape: {cosines.shape}')
			col = torch.arange(hidden_size+1).unsqueeze(0).repeat(le_batch_size, 1) 
			plt.scatter(cosines.detach().cpu(), all_norms_list.detach().cpu(), c = col)
			plt.title(f'Cosine Similarity vs Gradient Norm, p = {p}, epoch = {epoch}')
			plt.xlabel('Cosine Similarity')
			plt.ylabel('Grad Q Norms')
			plt.colorbar(label = 'Rank')
			plt.savefig(f'{plots_folder}/p{p}_e{epoch}_CosNormRelation.png', dpi = 400, bbox_inches = 'tight')
			plt.close()
			

			plt.figure(figsize = (8,4))
			plt.scatter((dV_losses-init_loss).detach().cpu(), all_norms_list.detach().cpu())
			plt.title(f'Loss Reduction vs Gradient Norm, p = {p}, epoch = {epoch}')
			plt.xlabel('Loss Reduction')
			plt.ylabel('Grad Q Norms')
			plt.savefig(f'{plots_folder}/p{p}_e{epoch}_LossNormRelation.png', dpi = 400, bbox_inches = 'tight')
			plt.close()
			
			plt.figure(figsize = (8,4))
			# print(f'Cosines shape: {cosines.shape}')
			plt.scatter(cosines.detach().cpu(), full_loss_diffs.detach().cpu(), c = 'k')
			plt.title(f'Cosine Similarity vs Loss (vs. Full), p = {p}, epoch = {epoch}')
			plt.xlabel('Cosine Similarity')
			plt.ylabel('Loss vs. Full rank')
			plt.savefig(f'{plots_folder}/p{p}_e{epoch}_CosLossRelation.png', dpi = 400, bbox_inches = 'tight')
			plt.close()
			
			plt.figure(figsize = (8,4))
			# print(f'Cosines shape: {cosines.shape}')
			col = torch.arange(le_batch_size).unsqueeze(0).repeat(1, hidden_size + 1) 
			plt.scatter(q_list.unsqueeze(1).repeat(1,le_batch_size), all_norms_list.detach().cpu(), c = col)
			plt.title(f'Gradient Norm vs. Rank, p = {p}, epoch = {epoch}')
			plt.xlabel('Rank')
			plt.ylabel('Gradient Norm')
			plt.savefig(f'{plots_folder}/p{p}_e{epoch}_RankNormRelation.png', dpi = 400, bbox_inches = 'tight')
			plt.close()
			
			plt.figure(figsize = (8,4))
			# print(f'Cosines shape: {cosines.shape}')
			col = ranks[1:].unsqueeze(1).repeat(1, le_batch_size) 
			# print(f'FTLE shape: {ftle_max_list[:, 1:].cpu().shape}')
			# print(f'Norms list shape: {all_norms_list[:,1:].shape}')
			print(f'Norms list shape: {col.shape}')
			plt.scatter(ftle_max_list[1:].cpu(), all_norms_list[1:].detach().cpu(), c = col)
			print(f'Max FTLEs: {ftle_max_list}')
			plt.title(f'Gradient Norm vs. Max FTLE, p = {p}, epoch = {epoch}')
			plt.xlabel('Max FTLE')
			plt.ylabel('Gradient Norm')
			plt.colorbar(label = 'Rank')
			plt.savefig(f'{plots_folder}/p{p}_e{epoch}_MaxFTLENormRelation.png', dpi = 400, bbox_inches = 'tight')
			plt.close()
			
			plt.figure(figsize = (8,4))
			# print(f'Cosines shape: {cosines.shape}')
			col = ranks[1:].unsqueeze(1).repeat(1, le_batch_size) 
			plt.scatter(ftle_means_list[1:].cpu(), all_norms_list[1:].detach().cpu(), c = col)
			plt.title(f'Gradient Norm vs. Mean FTLE, p = {p}, epoch = {epoch}')
			plt.xlabel('Mean FTLE')
			plt.ylabel('Gradient Norm')
			plt.colorbar(label = 'Rank')
			plt.savefig(f'{plots_folder}/p{p}_e{epoch}_MeanFTLENormRelation.png', dpi = 400, bbox_inches = 'tight')
			plt.close()