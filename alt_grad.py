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
from lyapunov import param_split, rnn_jac, gru_jac
from collections import OrderedDict
from copy import deepcopy


def grad_activation(fcon, model, h_t, x_t):
	param_list = param_split(model.rnn_layer.all_weights, bias = fcon.model.rnn_atts['bias'])
	if fcon.model.rnn_atts['bias']:
		W, U, b_i, b_h = param_list
		b = b_i + b_h
	else:
		W, U = param_list
		b = [torch.zeros(h_t.shape[1], model.gate_size).to(device)]
	a_list = torch.zeros(seq_length, le_batch_size, fcon.model.rnn_atts['hidden_size'])
	for t in range(seq_length):
		a = W[0]@x_t[:,t].unsqueeze(-1) + U[0]@h_t[t].unsqueeze(-1) + b[0].unsqueeze(-1)
		if fcon.model.model_type == 'gru':
			r_x = slice(0*hidden_size,1*hidden_size)
			z_x = slice(1*hidden_size,2*hidden_size)
			n_x = slice(2*hidden_size,3*hidden_size)
			# print(f'a shape: {a.shape}')
			# print(f'a_z shape: {a[:,z_x].shape}, h_t shape: {h_t[t].shape}')
			# print(f'1-a_z shape: {(1 - a[:,z_x]).shape}, a_n shape: {a[:,n_x].shape}')
			# print(f'product 1 shape: {(a[:, z_x]*h_t[t]).shape}')
			# print(f'product 2 shape: {((1 - a[:,z_x])*a[:,n_x]).shape}')
			a = torch.mul(a[:, z_x],h_t[t].unsqueeze(-1)) + torch.mul(1 - a[:,z_x],a[:,n_x])
		a_list[t] = a.squeeze()
	der = sech(a_list)**2
	return der


def jac_list(fcon, model, h_t, x_t):
	l = x_t.shape[1]
	print(f'h_t shape: {h_t.shape}')
	J_list = torch.zeros(l, le_batch_size, fcon.model.rnn_atts['hidden_size'], fcon.model.rnn_atts['hidden_size'])
	for r in range(l):
		if fcon.model.model_type == 'rnn':
			a = rnn_jac(model.rnn_layer.all_weights, h_t[:, r].unsqueeze(0), x_t[:, r].unsqueeze(1), bias = fcon.model.rnn_atts['bias'])
		elif fcon.model.model_type == 'gru':
			a = gru_jac(model.rnn_layer.all_weights, h_t[:, r].unsqueeze(0), x_t[:, r].unsqueeze(1), bias = fcon.model.rnn_atts['bias'])
		# print(a.shape)
		J_list[r] = a.transpose(-2,-1)
	return J_list


def output_grad(fcon, model, targets, logits):
	weight = model.fc.weight
	loss_grad = logits[:seq_length] - targets[:seq_length]
	weight_array = weight.t().unsqueeze(0).unsqueeze(0).repeat(seq_length, le_batch_size, 1, 1)
	out_grad = (weight_array@loss_grad.unsqueeze(-1)).squeeze(-1)
	return out_grad.cpu()


def jac_product(fcon, J_list, t, s, verbose = False):
	J_prod = torch.eye(fcon.model.rnn_atts['hidden_size']).unsqueeze(0).unsqueeze(0).repeat(s-t+1, le_batch_size, 1, 1)
	for r in range(1, s-t+1):
		if verbose:
			print(f'J prod shape: {J_prod[r-1].shape}')
			print(f'J list shape: {J_list[t+r-1].shape}')
		J_prod[r] = J_prod[r-1] @ J_list[t+r-1]
	return J_prod


def class_grad_h_summand(fcon, model, T, J_prod, out_grad):
	del_h_p = torch.zeros(T, le_batch_size, fcon.model.rnn_atts['hidden_size']).cpu()
	jp = J_prod[1:, 1:].cpu()
	for t in range(T):
		s = T-1
		# print(f'JP shape: {jp[t,s].shape}, out_grad shape: {out_grad.shape}')
		del_h_p[t] = torch.matmul(jp[t, s], out_grad[s].unsqueeze(-1)).squeeze()
	return del_h_p


def grad_h_summand(fcon, model, T, J_prod, out_grad):
	del_h_p = torch.zeros(T, T, le_batch_size, fcon.model.rnn_atts['hidden_size']).cpu()
	jp = J_prod[1:, 1:].cpu()
	for t in range(T):
		for s in range(t,T):
			del_h_p[t, s] = torch.matmul(jp[t, s], out_grad[s].unsqueeze(-1)).squeeze(-1)
	return del_h_p.sum(dim=1)


def grad_h_list(fcon, model, T, J_prod, targets, logits, clas = False):
	out_grad = output_grad(fcon, model, targets, logits)
	# print(f'out grad shape: {out_grad.shape}')
	if clas:
		x = class_grad_h_summand(fcon, model, T, J_prod, out_grad)
	else:
		x = grad_h_summand(fcon, model, T, J_prod, out_grad)
	if not clas:
		del_h = x + torch.sum(x, dim = 0, keepdims = True) - x.cumsum(dim = 0)
	else:
		del_h = x
	return del_h


def grad_V(fcon, model, grad_hList, h_in, x_t):
	phi_prime = grad_activation(fcon, model, h_in, x_t)
	test = grad_hList.unsqueeze(-1)[-1, 0]@h_in.cpu()[0,-1].unsqueeze(0)
	h_prod = (grad_hList.unsqueeze(-1)@h_in.cpu().unsqueeze(-2))
	W = model.fc.weight
	# print(f'Phi prim shape: {phi_prime.shape}, h_prod shape: {h_prod.shape}')
	gradV_summands = phi_prime@h_prod
	return gradV_summands.sum(dim = 0)
	
	
if __name__ == '__main__':
	#Configuration
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
	start_idx = 0
	seq_length = 112
	input_size = 7
	max_sub_epoch = 1
	in_epoch_saves = 4
	lr = 0.02
	fix_W = True
	overwrite = True
	calc_cosines = False
	plot_norms = False
	qs = True
	
	# params = torch.linspace(0.005, end= 0.025, steps = 2)
	params = [0.001, 0.005, 0.025, 0.05, 0.25, 0.5]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
	dcon = DataConfig('../Dataset/', input_size = input_size, batch_size= batch_size, input_seq_length = seq_length, 
												target_seq_length = 1, val_frac = 0.2, 
												test_frac = 0, name_params = {'insize':input_size}, download = True)
	mcon = ModelConfig(model_type, 1, hidden_size, dcon.input_size, output_size = output_size, dropout=dropout, 
						init_type = 'normal', init_params = {'mean':0, 'std':p},
						device = device, bias = False, id_init_param = 'std')                                            
	tcon = TrainConfig(model_dir = 'SMNIST/Models', batch_size = batch_size, max_epoch = max_epoch, 
														optimizer = 'sgd', learning_rate = learning_rate)
	fcon = FullConfig(dcon, tcon, mcon)
	
	le_input, le_target = torch.load('SMNIST/le_setup.p', map_location = device)
	# print(f'Input shape: {le_input.shape}')
	le_input = le_input.to(fcon.device).squeeze(1)[:, start_idx:start_idx+seq_length]
	le_target = le_target.to(fcon.device)
	model = RNNModel(fcon.model).to(fcon.device)
	ckpt = load_checkpoint(fcon, load_epoch = 0)
	model = ckpt[0]
	# model.rnn_layer.all_weights[0][1] = V + torch.Tensor([[0.2, 0, 0],[0, 0,0],[0,0,0]])
	weight_list = [model.fc.weight]
	if fix_W:
		optimizer = fcon.train.get_optimizer(model.rnn_layer.parameters())
	else: 
		optimizer = fcon.train.get_optimizer(model.parameters())
	criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
	cos = nn.CosineSimilarity(dim = 1)
	for g in optimizer.param_groups:
		g['lr'] = lr
	if in_epoch_saves >0:
		if os.path.exists('SMNIST/training_saveIdcs.p'):
			save_idcs = torch.load('SMNIST/training_saveIdcs.p', map_location = device)
		else:
			train_dataloader = torch.utils.data.DataLoader(fcon.data.datasets['train_set'], 
															batch_size = fcon.train.batch_size)
			epoch_samples = len(list(train_dataloader))
			save_idcs = part_equal(epoch_samples, in_epoch_saves)
			torch.save(save_idcs, 'SMNIST/training_saveIdcs.p')
	else:
		save_idcs = []
		
	# torch.autograd.set_detect_anomaly(True)
	#Iterate over all trained networks
	for p in params:
		p = float(int(p*1000))*1.0/1000
		fcon.model.init_params['std'] = p
		print(f'Parameter = {p}')
		for epoch in range(1, max_epoch+1, save_interval):
			print(f"Epoch {epoch}")
			for it in range(in_epoch_saves + 1):
				#Load Model Training Checkpoint
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
				pred_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_logits.p', map_location = device)
				print(f'gradV list shape: {gradV_list.shape}')
				

				for g in optimizer.param_groups:
					g['lr'] = lr
				ftle_dict = torch.load(f'SMNIST/LEs/{fcon.name()}_e{epoch}{suffix}_FTLE.p', map_location = device)
				h = ftle_dict['h'].to(device)
				
				model_state = model.state_dict()
				model_state2 = deepcopy(model.state_dict())
				#Calculate Gradients for comparison
				model.dropout = nn.Dropout(p = 0)
				V = model.rnn_layer.all_weights[0][1]
				U = model.rnn_layer.all_weights[0][0]
				W = model.fc.weight
				h_list = ftle_dict['h_list']
				x_list = ftle_dict['x_list']
				targets_ftle = ftle_dict['targets']
				le_input, le_target = (x_list, targets_ftle.to(fcon.device))
				FTLE_Jacs = ftle_dict['Jacs']
				# print(targets_ftle)
				
				h_le = ftle_dict['h'].to(fcon.device)
				# h_le = torch.zeros_like(h_le, requires_grad = True)
				outputs, h_t = model(x_list[:,:seq_length], h_le)
				h_t.retain_grad()
				h_in = torch.cat((h_le, h_t.transpose(0,1)), dim = 0)[:-1].detach()
				J_list = jac_list(fcon, model, h_in.transpose(0,1), x_list) #List of all the Jacobians for the sequences
				h = []
				h.append(h_le)
				for i in range(seq_length):
					out, h_new = model(x_list[:, i].unsqueeze(1), h[i])
					h.append(h_new.transpose(0,1))
					h[i+1].retain_grad()
				out, _ = model(x_list[:, seq_length-1].unsqueeze(1), h[seq_length-1])
				torch.set_printoptions(sci_mode = False, threshold = 10000)
				targets = f.one_hot(le_target, 10).unsqueeze(0).repeat(seq_length, 1, 1)
				preds = f.softmax(model.fc(h_t), dim = -1).transpose(0,1)
				loss = criterion(out, le_target)
				grads = []
				new_loss = []
				gradW_list_new = []
				gradV_list_new = []
				gradU_list_new = []
				init_V =V.data.clone().detach()
				init_W =W.data.clone().detach()
				init_U =U.data.clone().detach()

				print(optimizer)
				for i in range(le_batch_size):
					optimizer.zero_grad()
					out_update, h_n = model(le_input, h_le)
					loss_update = criterion(out_update, targets_ftle)
					loss_update[i].backward(retain_graph = False)
					gradW_list_new.append(W.grad)
					gradV_list_new.append(V.grad)
					gradU_list_new.append(U.grad)
					gradV_PT = V.grad.data
					optimizer.step()
					if i == le_batch_size -1:
						V_diff = V-init_V
						U_diff = U-init_U
						W_diff = W-init_W
						
					out_test, h_test = model(x_list[i].unsqueeze(0), h_le[:, i].unsqueeze(1))
					new_loss.append(criterion(out_test, le_target[i:i+1]).item())
					model.load_state_dict(model_state2)
					model_state2 = deepcopy(model_state)

				gradW_list_new= torch.stack(gradW_list_new)
				gradV_list_new= torch.stack(gradV_list_new)
				gradU_list_new= torch.stack(gradU_list_new)

				new_loss_name = f'SMNIST/Grads/p{p}_newLoss_e{epoch}{suffix}.p'
				torch.save(torch.Tensor(new_loss), new_loss_name)
				torch.save((gradV_list, gradU_list, gradW_list), f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_Newgrads.p')
				
				rvals = ftle_dict['rvals'].transpose(0,1).cpu() #List of r-values from FTLE calculation
				qvects = ftle_dict['qvects'].cpu() #List of Q-vectors from FTLE calculation
				ftle_jacs = ftle_dict['Jacs']
				Q0 = torch.eye(hidden_size).unsqueeze(0).unsqueeze(0).repeat(le_batch_size, 1, 1,1)
				qvec_list = torch.cat((Q0, qvects), dim = 1)
				s = ftle_dict['s'].cpu()
				
				q_jac = qvec_list[:, 1:]@rvals[:].transpose(0,1)@qvec_list[:, :-1].transpose(-2,-1)

				#The following tensors are meant to capture the respective products from t (first dim) 
				# to s (second dim), where both can range from 0 to seq_length
				r_prod = torch.zeros((seq_length+1, seq_length+1, le_batch_size, hidden_size, hidden_size)).cpu() #rvalues
				q_prod = torch.zeros((seq_length+1, seq_length+1, le_batch_size, hidden_size, hidden_size)).cpu() #full Q-product
				j_prod = torch.zeros((seq_length+1, seq_length+1, le_batch_size, hidden_size, hidden_size)).cpu() #Direct Jacobians
				prod_diff_norms = torch.zeros((seq_length+1, seq_length+1, le_batch_size))

				j_qr = (qvects.transpose(0,1)@rvals)
				q0 = torch.eye(hidden_size).unsqueeze(0).repeat(le_batch_size, 1, 1).unsqueeze(1)

				jp_fname = f'SMNIST/Grads/p{p}_jprod.p'
				overwrite_Jprod =True
				if os.path.exists(jp_fname) and not overwrite_Jprod:
					print('Loading JProd')
					j_prod = torch.load(jp_fname, map_location = device)
				else:
					for t in range(0, seq_length+1):
						j_prod[t, t:] = jac_product(fcon, J_list, t, seq_length)
					torch.save(j_prod, jp_fname)
				
				jq_prod = torch.zeros((seq_length+1, seq_length+1, le_batch_size, hidden_size, hidden_size)).cpu() #Direct Jacobians
				for t in range(0, seq_length+1):
					jq_prod[t, t:] = jac_product(fcon, q_jac.transpose(0,1), t, seq_length)
				
				if qs:
					q_prod_norms = []
					q_jac_norms = []
					gradV_norms=  []
					for q in range(0, hidden_size+1):
						qgrad_name =  f'SMNIST/Grads/p{p}_h{hidden_size}_gradVQR_red{q}.p'
						if not os.path.exists(qgrad_name) or overwrite:
							qp_fname = f'SMNIST/Grads/p{p}_h{hidden_size}_qprod_red{q}.p'
							if os.path.exists(qp_fname) and not overwrite:
								print(f'Loading Qvects for q = {q}')
								q_prod = torch.load(qp_fname, map_location = device)
							else:
								print(f'Calculating q = {q}')
								# q_jac = qvec_list[:, 1:, :, :-q]@rvals[:, :, :-q, :-q].transpose(0,1)@qvec_list[:, :-1, :, :-q].transpose(-2,-1)
								rvals_red = torch.zeros_like(rvals)
								rvals_red[:, :, :q,:q] =  rvals[:, :, :q, :q]
								q_jac = qvec_list[:, 1:]@rvals_red.transpose(0,1)@qvec_list[:, :-1].transpose(-2,-1)
								# q_prod = torch.zeros((seq_length+1, seq_length+1, le_batch_size, hidden_size, hidden_size)).cpu()
								for t in range(0, seq_length+1):
									q_prod[t, t:] = jac_product(fcon, q_jac.transpose(0,1), t, seq_length, verbose = False)
									
								q_jac_norms.append(torch.linalg.norm(q_jac, dim = (-2, -1)).mean())
								torch.save(q_prod, qp_fname)
								# prod_diff = j_prod - q_prod#find difference between two calculations
								# diff_norm = torch.linalg.norm(prod_diff, ord = 2, dim = (-1,-2))
							grad_h_QR = grad_h_list(fcon, model, seq_length, q_prod, targets, preds, clas = True)
							print(f'Q prod norm, Q = {q}: {torch.linalg.norm(grad_h_QR).detach()}')
							q_prod_norms.append(torch.linalg.norm(grad_h_QR).detach())
							gradV = grad_V(fcon, model, grad_h_QR, h_in, x_list)
							print(f'Grad V shape: {gradV.shape}')
							gradV_norms.append(torch.linalg.norm(gradV).detach())
							torch.save(gradV, qgrad_name)
							torch.save((q_jac_norms[-1], q_prod_norms[-1], gradV_norms[-1]), f'{qgrad_name.replace(".p", "")}_norms.p')							
						else:
							gradV = torch.load(qgrad_name, map_location = device)
							norms = torch.load(f'{qgrad_name.replace(".p", "")}_norms.p', map_location = device)
							for list, norm in zip([q_prod_norms, q_jac_norms, gradV_norms], norms):
								list.append(norm)
					q_prod_norms = torch.stack(q_prod_norms)
					q_jac_norms = torch.stack(q_jac_norms)
					gradV_norms = torch.stack(gradV_norms)
					print(f'Q Product Norms: {q_prod_norms}')
					# print(f'Q Product Norms: {q_prod_norms}')
						
					qgrad_name =  f'SMNIST/Grads/p{p}_h{hidden_size}_gradVQRFull.p'
					if not os.path.exists(qgrad_name) or overwrite:
						qp_fname = f'SMNIST/Grads/p{p}_h{hidden_size}_qprodFull.p'
						if os.path.exists(qp_fname) and not overwrite:
							print('Loading full Q product')
							q_prod = torch.load(qp_fname, map_location = device)
						else:
							q_jac = qvec_list[:, 1:]@rvals[:, :].transpose(0,1)@qvec_list[:, :-1].transpose(-2,-1)
							print('Calculating full Q product')
							for t in range(0, seq_length):
								q_prod[t, t:] = jac_product(fcon, q_jac.transpose(0,1), t, seq_length, verbose = False)
							torch.save(q_prod, qp_fname)
						grad_h_QR = grad_h_list(fcon, model, seq_length, q_prod, targets, preds, clas = True)
						gradV_QR = grad_V(fcon, model, grad_h_QR, h_in, x_list)
						torch.save(gradV_QR, qgrad_name)
					else:
						gradV_QR = torch.load(qgrad_name, map_location = device)
					
					start_inx = 0
					end_inx = 6
					prod_diff = j_prod - q_prod#find difference between two calculations
					diff_norm = torch.linalg.norm(prod_diff, ord = 2, dim = (-1,-2))
				
				grad_h_J = grad_h_list(fcon, model, seq_length, j_prod, targets, preds, clas = True)
				gradV_J = grad_V(fcon, model, grad_h_J, h_in, x_list)
				normalizer = torch.mean(torch.linalg.norm(gradV_J, dim = (-2,-1))).detach()

				plots_folder = f'SMNIST/Plots/p{p}' 
				if not os.path.exists(plots_folder):
					os.makedirs(plots_folder)
				diff_norms = []
				if calc_cosines:
					cosines = []
					for q in range(hidden_size+1):
						print(f'q = {q}')
						if q == hidden_size + 1:
							gradQ = gradV_QR
						else:
							gradQ = torch.load(f'SMNIST/Grads/p{p}_h{hidden_size}_gradVQR_red{q}.p')
						grad_diffs = gradV_J.to(device)-gradQ.to(device)
						diff_norms.append(torch.mean(torch.linalg.norm(grad_diffs, dim = (-2,-1))).detach())
						cosines.append(cos(gradV_list.reshape(le_batch_size, -1, 1), gradQ.reshape(le_batch_size, -1,1).to(device)))
					q_list = torch.arange(hidden_size+1)
					cosines = torch.stack(cosines).detach().cpu().squeeze()
					torch.save(cosines, f'SMNIST/Grads/p{p}_e{epoch}{suffix}_cosines.p')
					print(f'cosines shape: {cosines.shape}')
					plt.figure(figsize = (8,6))
					print(f'Q list shape: {q_list.unsqueeze(1).repeat(1,le_batch_size).shape}')
					plt.scatter(q_list.unsqueeze(1).repeat(1,le_batch_size), cosines.detach().cpu())
					plt.title(f'Cosine similarity with Full Gradient, p = {p}, epoch = {epoch}')
					plt.ylabel('Cosine Similarity')
					plt.xlabel('Grad Rank')
					plt.savefig(f'{plots_folder}/p{p}_e{epoch}CosineSimilarity.png', dpi = 400, bbox_inches = 'tight')
					plt.close()
				
				if plot_norms:
					plt.figure(figsize = (4,3))
					plt.scatter(q_list, gradV_norms, label = 'grad V norm')
					plt.scatter(q_list, q_prod_norms, label = 'grad H norm')
					plt.scatter(q_list, q_jac_norms, label = 'Q Jac norm')
					plt.xlabel('Reduced Rank')
					plt.ylabel('Norm')
					plt.legend()
					plt.title(f'Gradient Norms for p = {p}')
					plt.savefig(f'{plots_folder}/p{p}_e{epoch}_Grad_Norms.png', dpi = 400, bbox_inches = 'tight')
					plt.close()
					# print(torch.norm(grad_h_J - grad_h_QR, dim =(-1)).mean())
						