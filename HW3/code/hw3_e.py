import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [1,1/2,0,0,0,0],
    [0,1/2,1,1,0,0],
    [0,0,0,0,1,1],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0]
])

B = np.array([
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [1/3,1/3,1/3,1/3,1/3,1/3],
    [1/3,1/3,1/3,1/3,1/3,1/3],
    [1/3,1/3,1/3,1/3,1/3,1/3],
])
       
   
all_ranks = []
top_rank = []
for BETA in [0.01, 0.5, 0.99]:
    sum = BETA * A + (1 - BETA) * B

    eig_values, eig_vectors = np.linalg.eig(sum)
    eig_pairs = [
        ((eig_values[i]), eig_vectors[:,i])
        for i in range(len(eig_values))
    ]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)    

    ranks = eig_pairs[0][1]
    ranks = ranks / np.sum(ranks)
    print(BETA, ranks)
    
    top_rank.append(max(enumerate(ranks[:3]), key=lambda x: x[1])[0])
    
    all_ranks.append(ranks[:3])
    
print(list(set(top_rank)))
    
plt.plot( [0.01, 0.5, 0.99], all_ranks)
plt.show()
