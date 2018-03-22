import numpy as np
import random

fund_mats = np.load('fundamental_matrices.npy')
original_fund_mats = np.load('fundamental_matrices.npy')
percent = 0.01

for i in range(0, fund_mats.shape[1]):
  data_block_i = fund_mats[0,i]
  for j in range(data_block_i.shape[0]):
    for k in range(data_block_i.shape[1]):
      val = data_block_i[j,k]
      #print('-----------')
      #print(val)
      delta = abs(val)*percent
      #print(delta)
      new_val = random.uniform(val-delta,val+delta)
      #print(new_val)
      #print(val-new_val)
      fund_mats[0,i][j,k] = new_val

np.save('fundamental_matrices_lin_noise.npy', fund_mats)
diff = np.linalg.norm((original_fund_mats[0,0]-fund_mats[0,0]).flatten())
print(diff)
