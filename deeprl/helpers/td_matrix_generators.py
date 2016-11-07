import numpy as np


# relation and credit matrices for a DQN where action space size is n. n+1st node is value function,
# n+2nd node is reward function
def DQN_TD_matrix(n, discount_factor=.99):
    relations = np.zeros(shape=(n+2, n+1))
    for i in range(0, n+1):
        relations[0][i] = 1.
        relations[1][i] = discount_factor

    credit = np.ones(shape=(1,n))
    credit = np.concatenate((credit, np.eye(n, n)), axis=0)

    return relations, credit


# action conditional td net with full prediction tree. n=number of actions, l=depth of tree.
def action_conditional_tree(n, l=1):
    dim = 0
    for i in range(l+1):
        dim += n**i

    relations = np.zeros(shape=(dim, dim))
    for j in range(dim):
        if j > 0:
            relations[np.floor(float(j-1)/n)][j] = 1.

    identity = np.eye(n, n)
    credit = np.zeros(shape=(1,n))
    for k in range((dim-1)/n):
        credit = np.concatenate((credit, identity), axis=0)
    return relations, credit

# combining two question networks


