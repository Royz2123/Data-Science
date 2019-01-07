
import numpy as np
import matplotlib.pyplot as plt

EXAMPLE_A3 = np.array([
    [1,1/2,0,0,0,0],
    [0,1/2,1,1,0,0],
    [0,0,0,0,1,1],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0]
])

EXAMPLE_B3 = np.array([
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [1/3,1/3,1/3,1/3,1/3,1/3],
    [1/3,1/3,1/3,1/3,1/3,1/3],
    [1/3,1/3,1/3,1/3,1/3,1/3],
])


def create_A(N):
    A = []
    TELEPORTS = N * (N - 1) // 2
    TOTAL_NODES = N + N * (N - 1) // 2

    for row_index in range(N):
        row = [0] * TOTAL_NODES
        if row_index == 0:
            row[0] = 1
            row[1] = 1/2
        elif row_index == N - 2:
            row[row_index] = 1/2
            row[row_index + 1] = 1
        elif row_index == N - 1:
            pass
        else:
            row[row_index] = 1/2
            row[row_index + 1] = 1/2
            
        start = N + (row_index - 1) * row_index // 2
        end = start + row_index
        for index in range(start, end):
            row[index] = 1
        A.append(row)
    
    for row_index in range(N, TOTAL_NODES):
        row = [0] * TOTAL_NODES
        A.append(row)

    return np.array(A)
    
def create_B(N):
    B = []

    TELEPORTS = N * (N - 1) // 2
    TOTAL_NODES = N + N * (N - 1) // 2

    for row_index in range(N):
        row = [0] * TOTAL_NODES
        B.append(row)
        
    for row_index in range(TELEPORTS):
        row = [1 / TELEPORTS] * TOTAL_NODES
        B.append(row)
        
    return np.array(B)
  
N = 15
RESOLUTION = 100
  
A = create_A(N)
B = create_B(N)         
   
all_ranks = []
top_rank = []
for BETA in range(RESOLUTION + 1):
    BETA /= RESOLUTION
    sum = BETA * A + (1 - BETA) * B

    eig_values, eig_vectors = np.linalg.eig(sum)
    eig_pairs = [
        ((eig_values[i]), eig_vectors[:,i])
        for i in range(len(eig_values))
    ]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)    

    ranks = eig_pairs[0][1]
    ranks = ranks / np.sum(ranks)
    
    top_rank.append(max(enumerate(ranks[1:N]), key=lambda x: x[1])[0])
    
    all_ranks.append(ranks[1:N])
    
print("Found Nodes: ", list(set(top_rank)))
    
plt.plot(np.array(list(range(RESOLUTION + 1))) / RESOLUTION, all_ranks)
plt.title("Rank as a function of Beta")
plt.xlabel("Beta")
plt.ylabel("Rank")
plt.show()
