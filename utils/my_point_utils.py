# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def index_points(xyz, idx):
	"""
	xyz: points with size (B, C, N)
	idx: indexs used to select points from xyz, with size (B, S) or (B, S, n_neighbour)

	return:
	new_xyz (B, C, S) or (B, C, S, n_neighbour)
	"""
	if len(xyz.shape) == 4:
		xyz = xyz.squeeze()
	device = xyz.device
	B = xyz.shape[0] 
	view_shape = list(idx.shape)
	view_shape[1:] = [1] * (len(idx.shape)-1)
	repeat_shape = list(idx.shape)
	repeat_shape[0] = 1
	batch_idx = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

	xyz = xyz.permute(0,2,1)
	new_xyz = xyz[batch_idx, idx,:]
	if len(new_xyz.shape) == 3:
		new_xyz = new_xyz.permute(0, 2, 1)
	elif len(new_xyz.shape) == 4:
		new_xyz = new_xyz.permute(0, 3, 1, 2)

	return new_xyz

def farthest_point_sampling(xyz, npoints):
	"""
	xyz: points with size (B, C, N)
	npoint: (type int) number of centroids

	return:
	centroids (B, npoint)
	"""
	assert len(xyz.shape) == 3
	B, C, N = xyz.shape
	device = xyz.device
	centroids = torch.zeros(B, npoints, dtype=torch.long).to(device)
	distance = torch.ones(B, N).to(device) * 1e5
	farthest = torch.randint(0, N, size=(B,), dtype=torch.long).to(device)
	batch_idx = torch.arange(B, dtype=torch.long).to(device)
	for i in np.arange(npoints):
		centroids[:,i] = farthest
		center = xyz[batch_idx,:,farthest].unsqueeze(2)
		dist = torch.sum((xyz-center)**2, dim=1)
		mask = dist < distance
		distance[mask] = dist[mask]
		_, farthest = torch.max(distance, dim=1)

	return centroids

def square_distance(X, Z=None):
	"""
	X: point cloud tensor (B, 3, N)
	Z: another point cloud tensor (B, 3, M)
	when Z is none, function compute the pair-wise L2 distance of X
	when Z is not none, compute the pair-wise L2 distance between X and Z

	return:
	dist (B, N, M) or (B, N, N)
	"""
	B, C, N = X.shape
	if Z is None:
		Z = X 
		XZ = torch.matmul(X.permute(0,2,1), Z)
		XX = torch.sum(X*X, dim = 1).view(B, N, 1)
		ZZ = XX.permute(0, 2, 1) #(B, 1, N)
		return XX - 2*XZ + ZZ
	else:
		_, _, M = Z.shape
		XZ = torch.matmul(X.permute(0,2,1), Z)
		XX = torch.sum(X*X, dim = 1).view(B, N, 1)
		ZZ = torch.sum(Z*Z, dim = 1).view(B, 1, M)
		return XX - 2*XZ + ZZ

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, C, N]
        new_xyz: query points, [B, C, S]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, C, N = xyz.shape
    _, _, S = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)
    if radius is not None:
    	group_index = torch.arange(N, dtype=torch.long).to(device).view(1,1,N).repeat([B, S, 1])
    	group_index[sqrdists > radius**2] = N
    	group_index = group_index.sort(dim=-1)[0][:,:, :nsample]
    	mask = (group_index==N)
    	group_index_first = group_index[:,:,0].view(B,S,1).repeat([1,1,nsample])
    	group_index[mask] = group_index_first[mask]
    else:
    	group_index = torch.sort(sqrdists, dim=-1)[1][:,:, :nsample]

    return group_index	



def main():
	test_data = torch.randn(8, 3, 100)
	test_index = torch.randint(low=0, high=6, size=(8, 5, 16))
	select_point = index_points(test_data, test_index)
	print(select_point.shape)
	#print(loss.item())

if __name__ == '__main__':
	main()