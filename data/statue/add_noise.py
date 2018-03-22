import numpy as np
import random

dense_matches = np.load('dense_matches.npy')
original_matches = np.load('dense_matches.npy')
percent = 0.3

for i in range(0, dense_matches.shape[0]):
  data_block_i = dense_matches[i]
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
      dense_matches[i][j,k] = new_val

np.save('dense_matches_lin_noise30.npy', dense_matches)
diff = np.linalg.norm((original_matches[0]-dense_matches[0]).flatten())
print(diff)
