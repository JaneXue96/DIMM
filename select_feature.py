import numpy as np
import os
# import heapq


results_dir = './outputs/single_task'
tasks = ['5849', '4019', '25000', '41401']
model = 'DIMM'


def sort_features(f_matrix):
    in_dim, out_dim = f_matrix.shape[0], f_matrix.shape[1]
    matrix_sum = np.sum(np.abs(f_matrix), axis=1)
    # idx_sum = map(matrix_sum.index, heapq.nlargest(20, matrix_sum))
    idx_sum = matrix_sum.argsort()[::-1][0:100].tolist()

    # matrix_avg = matrix_sum / out_dim
    # idx_avg = map(matrix_avg.index, heapq.nlargest(20, matrix_avg))
    # idx_avg = matrix_avg.argsort()[::-1][0:20].tolist()

    # return np.concatenate((idx_sum, idx_avg), axis=0)
    return np.array([idx_sum])


for task in tasks:
    path = os.path.join(results_dir, task, model, 'results')
    index_name = os.path.join(path, task + '_index_W.txt')
    medicine_name = os.path.join(path, task + '_medicine_W.txt')
    index = np.loadtxt(index_name, delimiter=',')
    medicine = np.loadtxt(medicine_name, delimiter=',')
    idx_index = sort_features(index)
    idx_medicine = sort_features(medicine)

    np.savetxt(os.path.join(path, task + '_max_index.txt'), idx_index, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(path, task + '_max_medicine.txt'), idx_medicine, fmt='%d', delimiter=',')

