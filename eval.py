import spconv
import spconv.pool as spconvpool
from torch import nn
import torch
import numpy as np 
import pickle
import random
import csv
import time

class PokeNet(nn.Module):
	def __init__(self, shape):
		super().__init__()
		
		
		self.net = spconv.SparseSequential(            
			spconv.SparseConv3d(1, 32, 4),
			nn.LeakyReLU(),
			spconv.SparseConv3d(32, 32, 4),
			nn.LeakyReLU(),
			spconvpool.SparseMaxPool(3, 5),
			
			spconv.SparseConv3d(32, 32, 4),
			nn.LeakyReLU(),
			spconv.SparseConv3d(32, 32, 4),
			nn.LeakyReLU(),
			spconvpool.SparseMaxPool(3, 5),

			spconv.SparseConv3d(32, 32, 4),
			nn.LeakyReLU(),
			spconv.SparseConv3d(32, 32, 4),
			nn.LeakyReLU(),
			spconv.SparseConv3d(32, 1, 4),
			nn.LeakyReLU(),
			spconvpool.SparseMaxPool(3, 5),
			
			spconv.ToDense(),
			
			nn.Flatten(),
			
			nn.Linear(14688, 1000),
			nn.LeakyReLU(),
			nn.Linear(1000, 1000),
			nn.LeakyReLU(),
			nn.Linear(1000, 1),
			nn.LeakyReLU(),
#             nn.Sigmoid()
		)        
		self.shape = shape
		
	def forward(self, features, coors, batch_size):
		coors = coors.int() # unlike torch, this library only accept int coordinates.
		x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
		x = self.net(x)# .dense()
		return x


def determine_voxel_idx(x, y, z, min_x=-18.5, min_y=-13.5, min_z=-11, width=0.5):
	# Determine which voxel indices this atom belongs in.
	# 0 1 2
	# Return: list (x_idx, y_idx, z_idx)
	# 23.283158494668292 14.716941870547583 11.091082517990833
	# -18.417849987869186 -13.38094987292965 -10.673369985583971
	return [int((dim-min_dim) / width) for dim, min_dim in zip([x, y, z], [min_x, min_y, min_z])] 

def get_input_dict(batch):
	molecule_data = {'features': [], 'features_dense': [], 'indices': []}
	
	labels = []
	
	for idx, molecule in enumerate(batch):
		
		labels.append([int(molecule['label'])])

		for atomicNum, charge, coord in zip(molecule['atomicNum'], molecule['formalCharge'], molecule['3DConfig']):
			feature = atom_val_map[(atomicNum, charge)]
			voxel_idx = determine_voxel_idx(*coord)
			molecule_data['features'].append([feature])
			molecule_data['indices'].append(voxel_idx+[idx])
#             molecule_grid[tuple(voxel_idx)] = feature

	molecule_data['features'] = np.array(molecule_data['features'])
	molecule_data['indices'] = np.array(molecule_data['indices'])
	
	return molecule_data, labels

def shuffle_data(data):
	last_smiles = data[0]['smiles']
	output = []
	tmp = []
	for idx, molecule in enumerate(data):
		if last_smiles == molecule['smiles']:
			tmp.append(molecule)
		else:
			output.append(tmp)
			tmp = []
		last_smiles = molecule['smiles']
	random.shuffle(output)
	return [item for sublist in output for item in sublist]


def train_model(dataset, batch_size, log_file="misc_log.csv"):

	for i in range(0, len(dataset), batch_size):
		batch_dict, labels = get_input_dict(dataset[i:i + batch_size])
		
		actual_batch_size = len(dataset[i:i + batch_size])

		features = np.ascontiguousarray(batch_dict["features"]).astype(np.float32)
		indices = np.ascontiguousarray(batch_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)

		# Send to GPU
		indices_t = torch.from_numpy(indices).int().to(device).float()
		features_t = torch.from_numpy(features).to(device).float()
		
		
		labels_t = torch.FloatTensor(labels).to(device)
		
		y_pred = net(features_t, indices_t, actual_batch_size)
		
		return y_pred, labels_t


if __name__ == '__main__':
	test_set_path = "test_validation.p"

	dataset = pickle.load(open(test_set_path, "rb" ))

	device = torch.device('cuda')
	VOXEL_DIMS = [84, 57, 45]
	batch_size = 8

	net = PokeNet(VOXEL_DIMS).to(device).float()

	atom_val_map = {(1, 0): 1,
					(1, 1): 2,
					(5, 0): 3,
					(6, -1): 4,
					(6, 0): 5,
					(7, -1): 6,
					(7, 0): 7,
					(7, 1): 8,
					(8, -1): 9,
					(8, 0): 10,
					(8, 1): 11,
					(9, 0): 12,
					(11, 0): 13,
					(11, 1): 14,
					(15, 0): 15,
					(16, 0): 16,
					(16, 1): 17,
					(17, -1): 18,
					(17, 0): 19,
					(20, 2): 20,
					(35, -1): 21,
					(35, 0): 22,
					(53, 0): 23}

	# Shuffle the dataset
	random.shuffle(dataset)

	print(eval_model(dataset, batch_size))