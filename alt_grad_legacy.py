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
from lyapunov import param_split, rnn_jac
from collections import OrderedDict
   
def grad_activation(fcon, model, h_t, x_t):
	param_list = param_split(model.rnn_layer.all_weights, bias = fcon.model.rnn_atts['bias'])
	if fcon.model.rnn_atts['bias']:
		W, U, b_i, b_h = param_list
		b = b_i + b_h
	else:
		W, U = param_list
		b = [torch.zeros(h_t.shape[1], fcon.model.rnn_atts['hidden_size']).to(device)]
	a_list = torch.zeros(seq_length, le_batch_size, hidden_size)
	# print(f'x_t shape: {x_t.shape}')
	for t in range(seq_length):
	    a = W[0]@x_t[:,t].unsqueeze(-1) + U[0]@h_t[t].unsqueeze(-1) + b[0].unsqueeze(-1)
	    a_list[t] = a.squeeze()
	# one = torch.eye(hidden_size).unsqueeze(0).unsqueeze(0).repeat(seq_length, le_batch_size,1,1)
	# der =  one - tanh(a_list)**2
	der = sech(a_list)**2
	# print(f'Activations: {a_list}')
	# print(f'x_t: {x_t}')
	# print(f'Grad Activations: {der}')
	# print(f'Grad Activations shape: {der.shape}')
	return der
	
def jac_list(fcon, model, h_t, x_t):
	l = x_t.shape[1]
	print(f'h_t shape: {h_t.shape}')
	J_list = torch.zeros(l, le_batch_size, fcon.model.rnn_atts['hidden_size'], fcon.model.rnn_atts['hidden_size'])
	for r in range(l):
		a = rnn_jac(model.rnn_layer.all_weights, h_t[:, r].unsqueeze(0), x_t[:, r].unsqueeze(1), bias = fcon.model.rnn_atts['bias'])
		# print(a.shape)
		J_list[r] = a.transpose(-2,-1)
	return J_list

def output_grad(fcon, model, targets, logits):
	weight = model.fc.weight
	# print(f'Logits: {logits[0, 0]}')
	# print(f'Target: {targets[0, 0]}')
	loss_grad = logits[:seq_length] - targets[:seq_length]
	# print(f'Loss Grad: {loss_grad[0,0]}')
	# print(f'Output weight: {weight}')
	weight_array = weight.t().unsqueeze(0).unsqueeze(0).repeat(seq_length, le_batch_size, 1, 1)
	out_grad = (weight_array@loss_grad.unsqueeze(-1)).squeeze(-1)
	# print(f'Out grad shape: {out_grad.shape}')
	# print(f'out grad: {out_grad[:,0]}')
	return out_grad.cpu()

def jac_product(fcon, J_list, t, s):
	J_prod = torch.eye(fcon.model.rnn_atts['hidden_size']).unsqueeze(0).unsqueeze(0).repeat(s-t+1, le_batch_size, 1, 1)
	# print('Calculating Jprod')
	for r in range(1, s-t+1):
		# print(f'J List Element: {J_list[t+r-1, 0]}')
		# print(f'Prev J Product: {J_prod[r-1, 0]}')
		# J_prod[r] = J_prod[r-1] @ J_list[t+r-1].transpose(-2, -1)
		J_prod[r] = J_prod[r-1] @ J_list[t+r-1]
		# .transpose(-2, -1)
		# print(f'J prod element: {J_prod[r]}')
		# J_prod[r] = J_prod[r-1] @ J_list[t+r-1]
	# print(f'Output J_prod: {J_prod[:, 0]}')
	return J_prod

def class_grad_h_summand(fcon, model, T, J_prod, out_grad):
	del_h_p = torch.zeros(T, le_batch_size, fcon.model.rnn_atts['hidden_size']).cpu()
	# print(f'Class JP shape: {J_prod.shape}')
	# print(f'Full Jprod: {J_prod}')
	jp = J_prod[1:, 1:].cpu()
	for t in range(T):
		# print(f't = {t}')
		s = T-1
		# print(f'JP {s}, {t}: {jp[t,s,0]}')
		del_h_p[t] = torch.matmul(jp[t, s], out_grad[s].unsqueeze(-1)).squeeze()
		# print(f'Alt MatProd: {del_h_p[t]}')
	# print(f'del h shape: {del_h_p.shape}')
	# print(f'del_h_p: {del_h_p[:,0]}')
	return del_h_p

def grad_h_summand(fcon, model, T, J_prod, out_grad):
	del_h_p = torch.zeros(T, T, le_batch_size, fcon.model.rnn_atts['hidden_size']).cpu()
	# print(f'J_prod shape: {J_prod.shape}')
	jp = J_prod[1:, 1:].cpu()
	# print(f'jp sample: {jp[1,1,0,:5,:5]}')
	for t in range(T):
		for s in range(t,T):
			del_h_p[t, s] = torch.matmul(jp[t, s], out_grad[s].unsqueeze(-1)).squeeze(-1)
	return del_h_p.sum(dim=1)

def grad_h_list(fcon, model, T, J_prod, targets, logits, clas = False):
	out_grad = output_grad(fcon, model, targets, logits)
	# print(f'Full J Prod: {J_prod[:, :, 0]}')
	# print(f'J x out grad: {(J_list.transpose(-2,-1) @ out_grad[-1].unsqueeze(-1))[:, 0]}')
	# print(f'Out grads: {out_grad}')
	if clas:
		x = class_grad_h_summand(fcon, model, T, J_prod, out_grad)
	else:
		x = grad_h_summand(fcon, model, T, J_prod, out_grad)
	# print(f'Summands Shape: {x.shape}')
	if not clas:
		del_h = x + torch.sum(x, dim = 0, keepdims = True) - x.cumsum(dim = 0)
	else:
		del_h = x
	# print(f'X list: {x[:,0]}')
	# print(f'Del H: \n{del_h[:, 0, :]}')
	return del_h
	
def grad_V(fcon, model, grad_hList, h_in, x_t):
	phi_prime = grad_activation(fcon, model, h_in, x_t)
	# print(f'Phi Prime \n {phi_prime[:, 0]}')
	# print(f'Grad H List: {grad_h_list}')
	# print(f'Phi product: {phi_prime@grad_h_list.unsqueeze(-1)}')
	# print(f'Phi shape: {phi_prime.shape}')
	# print(f'Grad H List shape: {grad_hList.shape}')
	# print(f'H shape: {h_in.cpu().transpose(0,1).unsqueeze(-2).shape}')
	# print(f'H in shape: {h_in.shape}')
	# print(f'grad_hList shape: {grad_hList.shape}')
	test = grad_hList.unsqueeze(-1)[-1, 0]@h_in.cpu()[0,-1].unsqueeze(0)
	# print(f'test1 shape: {grad_hList.unsqueeze(-1).shape}')
	# print(f'test2 shape: {h_in.cpu().transpose(0,1).unsqueeze(-2).shape}')
	h_prod = (grad_hList.unsqueeze(-1)@h_in.cpu().unsqueeze(-2))
	# print(f'H product: \n {h_prod[:,0]}')
	# print(f'Test Product: \n {test}')
	# print(f'Expected Diagonals: {}')
	W = model.fc.weight
	gradV_summands = phi_prime@h_prod
	# print(f'gradV summands: \n {gradV_summands[:,0]}')
	# print(f'gradV summands shape: {gradV_summands.shape}')
	# print(f'Full grad V: {gradV_summands.sum(dim = 0)}')
	return gradV_summands.sum(dim = 0)
	
	
if __name__ == '__main__':
	#Configuration
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	batch_size = 128
	le_batch_size = 25
	output_size = 10
	max_epoch = 1
	learning_rate = 0.002
	dropout = 0.1
	hidden_size = 3
	save_interval = 1
	model_type = 'rnn'
	p = 0.001
	start_idx = 0
	seq_length = 28
	input_size = 28
	max_sub_epoch = 1
	in_epoch_saves = 4
	lr = 0.1
	overwrite = True
	qs = True
	
	params = torch.linspace(0.005, end= 0.025, steps = 5)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
	dcon = DataConfig('../Dataset/', input_size = input_size, batch_size= batch_size, input_seq_length = seq_length, 
												target_seq_length = 1, val_frac = 0.2, 
												test_frac = 0, name_params = {'insize':input_size}, download = True)
	mcon = ModelConfig(model_type, 1, hidden_size, dcon.input_size, output_size = output_size, dropout=dropout, 
						init_type = 'normal', init_params = {'mean':0, 'std':p},
						device = device, bias = False, id_init_param = 'std')                                            
	tcon = TrainConfig(model_dir = 'SMNIST/Models', batch_size = batch_size, max_epoch = max_epoch, 
														optimizer = 'adam', learning_rate = learning_rate)
	fcon = FullConfig(dcon, tcon, mcon)
	
	le_input, le_target = torch.load('SMNIST/le_setup.p', map_location = device)
	# print(f'Input shape: {le_input.shape}')
	le_input = le_input.to(fcon.device).squeeze(1)[:, start_idx:start_idx+seq_length]
	le_target = le_target.to(fcon.device)
	model = RNNModel(fcon.model).to(fcon.device)
	ckpt = load_checkpoint(fcon, load_epoch = 0)
	model = ckpt[0]
	# model.rnn_layer.all_weights[0][1] = V + torch.Tensor([[0.2, 0, 0],[0, 0,0],[0,0,0]])
	# print(f'V: {V}')
	weight_list = [model.fc.weight]
	optimizer = fcon.train.get_optimizer(model.parameters())
	criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
	
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
	for p in params[:1]:
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
				gradV_list, gradU_list, gradW_list, loss_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_grads.p', map_location = device)
				pred_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_logits.p', map_location = device)
				
				ftle_dict = torch.load(f'SMNIST/LEs/{fcon.name()}_e{epoch}{suffix}_FTLE.p', map_location = device)
				h = ftle_dict['h'].to(device)
				
				#Calculate Gradients
				model.dropout = nn.Dropout(p = 0)
				V = model.rnn_layer.all_weights[0][1]
				h_list = ftle_dict['h_list']
				x_list = ftle_dict['x_list']
				
				h_le = ftle_dict['h'].to(fcon.device)
				# h_le = torch.zeros_like(h_le, requires_grad = True)
				outputs, h_t = model(x_list[:,:seq_length], h_le)
				# print(f'h diff: {h_list.transpose(0,1)[0] - h_t[0]}')
				h_t.retain_grad()
				h_in = torch.cat((h_le, h_t.transpose(0,1)), dim = 0)[:-1].detach()
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
				grads =  loss[0].backward(retain_graph = True)
				gradV_PT = V.grad
				# print(f'gradV shape: {gradV_PT.shape}')
				
				
				J_list = jac_list(fcon, model, h_in.transpose(0,1), x_list) #List of all the Jacobians for the sequences
				# print(f'J list shape: {J_list.shape}')
				# print(f'J: {J_list[:, 0]}')
				rvals = ftle_dict['rvals'].transpose(0,1).cpu() #List of r-values from FTLE calculation
				qvects = ftle_dict['qvects'].cpu() #List of Q-vectors from FTLE calculation
				ftle_jacs = ftle_dict['Jacs']
				Q0 = torch.eye(hidden_size).unsqueeze(0).unsqueeze(0).repeat(le_batch_size, 1, 1,1)
				# print(f'Q Vects shape: {qvects.shape}')
				qvec_list = torch.cat((Q0, qvects), dim = 1)
				print(f'Qvec list shape: {qvec_list.shape}')
				# print(f'J list shape: {J_list.shape}')
				# print(f'FTLE J list shape: {ftle_jacs.shape}')
				s = ftle_dict['s'].cpu()
				
				q_jac = qvec_list[:, 1:]@rvals[:].transpose(0,1)@qvec_list[:, :-1].transpose(-2,-1)
				# print(f'Jac List: {J_list[0]}')
				# print(f'Q jac: {q_jac[0].transpose(-2,-1)}')
				print(f'Jac Diff {torch.linalg.norm(J_list[0]-q_jac[:, 0])}')
								
				#The following tensors are meant to capture the respective products from t (first dim) 
				# to s (second dim), where both can range from 0 to seq_length
				r_prod = torch.zeros((seq_length+1, seq_length+1, le_batch_size, hidden_size, hidden_size)).cpu() #rvalues
				q_prod = torch.zeros((seq_length+1, seq_length+1, le_batch_size, hidden_size, hidden_size)).cpu() #full Q-product
				j_prod = torch.zeros((seq_length+1, seq_length+1, le_batch_size, hidden_size, hidden_size)).cpu() #Direct Jacobians
				prod_diff_norms = torch.zeros((seq_length+1, seq_length+1, le_batch_size))

				j_qr = (qvects.transpose(0,1)@rvals)
				q0 = torch.eye(hidden_size).unsqueeze(0).repeat(le_batch_size, 1, 1).unsqueeze(1)
				
				# print('Diff between Jac and QR products:')
				jp_fname = f'SMNIST/Grads/p{p}_jprod.p'
				overwrite_Jprod =True
				if os.path.exists(jp_fname) and not overwrite_Jprod:
					print('Loading JProd')
					j_prod = torch.load(jp_fname, map_location = device)
				else:
					for t in range(0, seq_length+1):
						# print(f'J prod for t = {t} of {seq_length}')
						j_prod[t, t:] = jac_product(fcon, J_list, t, seq_length)
					torch.save(j_prod, jp_fname)
				
				jq_prod = torch.zeros((seq_length+1, seq_length+1, le_batch_size, hidden_size, hidden_size)).cpu() #Direct Jacobians
				for t in range(0, seq_length+1):
					# print(f'J prod for t = {t} of {seq_length}')
					jq_prod[t, t:] = jac_product(fcon, q_jac.transpose(0,1), t, seq_length)
				
				print(f'Jac prod Diff: {torch.linalg.norm(j_prod-jq_prod)}')
				if qs:
					for q in range(1, hidden_size):
						qgrad_name =  f'SMNIST/Grads/p{p}_h{hidden_size}_gradVQR_red{q}.p'
						if not os.path.exists(qgrad_name) or overwrite:
							qp_fname = f'SMNIST/Grads/p{p}_h{hidden_size}_qprod_red{q}.p'
							if os.path.exists(qp_fname) and not overwrite:
								print(f'Loading Qvects for q = {q}')
								q_prod = torch.load(qp_fname, map_location = device)
							else:
								print(f'Calculating q = {q}')
								for t in range(0, seq_length):
									r_prod[t,t] = torch.eye(hidden_size).unsqueeze(0).repeat(le_batch_size, 1, 1)
									for s in range(t, seq_length+1):
										q_red = torch.zeros_like(qvec_list)
										q_red[:, :, :-q] = qvec_list[:,:,:-q]
										# q_jac = q_red[:, 1:]@rvals[:].transpose(0,1)@qred[:, :-1].transpose(-2,-1)
										if s == t: #when s = t, the product is empty and is replaced with the identity
											q_prod[t,s] = torch.eye(fcon.model.rnn_atts['hidden_size']).unsqueeze(0).repeat(le_batch_size, 1, 1)
										else: #when s>=t+1, we multiply the product for s-1 by the current r-value (indexed by s-1 in rvals object, since it has no value for t = 0)
											r_prod[t, s] = r_prod[t, s-1]@(rvals[s-1].transpose(-1, -2))
											# if t == 0:
												# q_prev = torch.eye(hidden_size).unsqueeze(0).repeat(le_batch_size, 1, 1)
											# else:
												# q_prev =  q_red[:,t-1]
											# q_jac = qvec_list[:, 1:]@rvals[:].transpose(0,1)@qvec_list[:, :-1].transpose(-2,-1)
											q_prod[t, s] = q_red[:,t]@r_prod[t, s]@q_red[:,s].transpose(-2,-1)
								torch.save(q_prod, qp_fname)
								# prod_diff = j_prod - q_prod#find difference between two calculations
								# diff_norm = torch.linalg.norm(prod_diff, ord = 2, dim = (-1,-2))
							grad_h_QR = grad_h_list(fcon, model, seq_length, q_prod, targets, preds, clas = True)
							gradV = grad_V(fcon, model, grad_h_QR, h_in, x_list)
							torch.save(gradV, qgrad_name)
						else:
							gradV = torch.load(qgrad_name, map_location = device)
						
					qgrad_name =  f'SMNIST/Grads/p{p}_h{hidden_size}_gradVQRFull.p'
					if not os.path.exists(qgrad_name) or overwrite:
						qp_fname = f'SMNIST/Grads/p{p}_h{hidden_size}_qprodFull.p'
						if os.path.exists(qp_fname) and not overwrite:
							print('Loading full Q product')
							q_prod = torch.load(qp_fname, map_location = device)
						else:
							print('Calculating full Q product')
							for t in range(0, seq_length):
								# print(f't = {t}')
								r_prod[t,t] = torch.eye(hidden_size).unsqueeze(0).repeat(le_batch_size, 1, 1)
								for s in range(t, seq_length+1):
									# if s == t: #when s = t, the product is empty and is replaced with the identity
										# q_prod[t,s] = torch.eye(fcon.model.rnn_atts['hidden_size']).unsqueeze(0).repeat(le_batch_size, 1, 1)
									# else: #when s>=t+1, we multiply the product for s-1 by the current r-value 
														  # (indexed by s-1 in rvals object, since it has no value for t = 0)
									if s>t:
										r_prod[t, s] = r_prod[t, s-1]@(rvals[s-1].transpose(-1, -2))
										# r_prod[t, s] = r_prod[t, s-1]@(rvals[s-1])
									# r_prod[t, s] = ((rvals[s-1])@(r_prod[t, s-1])).transpose(-1, -2)
									q_prod[t, s] = (qvec_list[:, t]@r_prod[t, s]@qvec_list[:,s].transpose(-2,-1)).transpose(-2,-1)
									# print(f'Int Product: {(r_prod[t, s]@qvects[:,s-1].transpose(-2,-1))[0]}')
							torch.save(q_prod, qp_fname)
						grad_h_QR = grad_h_list(fcon, model, seq_length, q_prod, targets, preds)
						gradV_QR = grad_V(fcon, model, grad_h_QR, h_in, x_list)
						torch.save(gradV_QR, qgrad_name)
					else:
						gradV_QR = torch.load(qgrad_name, map_location = device)
					
					start_inx = 2
					end_inx = 5
					prod_diff = j_prod - q_prod#find difference between two calculations
					diff_norm = torch.linalg.norm(prod_diff, ord = 2, dim = (-1,-2))
					print(f'Product Difference: {prod_diff[start_inx,start_inx:end_inx,0]}')
					print(f'Q prod: {q_prod[start_inx, start_inx:end_inx, 0]}')
					# print(f'Diff quotient shape: {prod_diff.div(q_prod)[start_inx, start_inx:end_inx, 0]}')
					# print(f'R prod shape: {r_prod[0, :, 0]}')
				
				grad_h_J = grad_h_list(fcon, model, seq_length, j_prod, targets, preds, clas = True)
				# print(f'out grad shape: {out_grad.shape}')
				# print(f'Jacobian: {J_list[1,0]}')
				# print(f'J prod: {j_prod[1,2, 0]}')
				# print(f'Middle Del H: {j_prod[1,2, 0]@out_grad[-1, 0]}')
				# phi_prime = grad_activation(fcon, model, h_in[:, :-1], le_input)
				# test = phi_prime@grad_h_J.unsqueeze(-1)@h_t.cpu().transpose(0,1).unsqueeze(-2)
				gradV_J = grad_V(fcon, model, grad_h_J, h_in, x_list)
				# print(f'Grad V List shape: {gradV_list.shape}')
				
				# print(f'Grad V: \n {gradV_J[0]}')
				# print(f'h_t: {h_t[0]}')
				# print(f'GradH quotient: {gradh_PT.div(del_h[:, 0, :])}')
				# print(f'Pytorch Grad V list shape: {gradV_list.shape}')
				# print(f'Pytorch Grad: \n {gradV_PT}')
				# print(f'Quotient: \n{gradV_PT.div(gradV_J[0])}')
				# print(f'Grad Diff: \n {(gradV_PT.cpu()-gradV_J[0])}')
				# print(f'Percentage Diff: \n {torch.div((gradV_PT.cpu()-gradV_J[0]), gradV_PT.cpu())}')
				normalizer = torch.mean(torch.linalg.norm(gradV_J, dim = (-2,-1))).detach()
				# print(normalizer)
				
				diff_norms = []
				for q in range(hidden_size):
					print(f'q = {q}')
					if q == 0:
						gradQ = gradV_QR
					else:
						gradQ = torch.load(f'SMNIST/Grads/p{p}_h{hidden_size}_gradVQR_red{q}.p')
					diff_norms.append(torch.mean(torch.linalg.norm(gradV_J.to(device)-gradQ.to(device), dim = (-2,-1))).detach())
				q_list = list(range(hidden_size))
				print(diff_norms)
				# plt.scatter(q_list, diff_norms)
				plt.xlabel('Dimension Reduction')
				plt.ylabel('Mean Difference Fr. Norm')
				plt.savefig('SMNIST/Plots/Grad_diffs.png')
				
				# print(torch.norm(grad_h_J - grad_h_QR, dim =(-1)).mean())
						