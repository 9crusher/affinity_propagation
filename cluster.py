import numpy as np


def similarity(features_matrix):
    '''
    Accepts a matrix with column 0 containing the labels
    of points to group and the rest of the columns containing
    feature data
    '''
    in_shape = features_matrix.shape
    sim_matrix = np.zeros((in_shape[0], in_shape[0]))
    for i in range(in_shape[0]):
        for j in range(in_shape[0]):
            for in_col in range(in_shape[1]): # skip id column
                sim_matrix[i,j] += (features_matrix[i][in_col] - features_matrix[j][in_col]) ** 2
    sim_matrix += np.diag(np.repeat(np.amax(sim_matrix.flatten()), in_shape[0])) #Affects number of clusters
    return sim_matrix * -1

def responsibility(similairity_matrix):
    '''
    Calculate responsibility matrix given a similairity matrix.
    '''
    sim_shape = similairity_matrix.shape
    responsibility_matrix = np.zeros(sim_shape)
    for i in range(sim_shape[0]):
        for j in range(sim_shape[1]):
            responsibility_matrix[i,j] = similairity_matrix[i,j] - np.amax(np.delete(similairity_matrix[i], j))
    return responsibility_matrix

def availability(responsibility_matrix):
    '''
    Calc availability from responsibility
    '''
    availability_matrix = np.zeros(responsibility_matrix.shape)
    for i in range(availability_matrix.shape[0]):
        for j in range(availability_matrix.shape[1]):
            col_no_i = np.delete(responsibility_matrix[:, j].flatten(), i)
            col_sum = np.sum(col_no_i[col_no_i > 0])
            availability_matrix[i, j] = min(0, responsibility_matrix[j,j] + col_sum)
    # Diag
    for i in range(availability_matrix.shape[0]):
        col_without_current = np.delete(responsibility_matrix[:, i].flatten(), i)
        availability_matrix[i, i] = np.sum(col_without_current[col_without_current > 0])
    return availability_matrix

def criterion(responsibility_matrix, availability_matrix):
    return responsibility_matrix + availability_matrix







inp = np.array([[3, 4, 3, 2 ,1 ], [4, 3, 5, 1, 1], [3, 5, 3, 3, 3], [2, 1, 3, 3, 2], [1, 1, 3, 2, 3]])
sim = similarity(inp)
resp = responsibility(sim)
avail = availability(resp)
crit = criterion(resp, avail)
exemplars = np.amax(crit, axis=1)
groups = dict()
for i in range(len(exemplars)):
    if exemplars[i] in groups:
        groups[exemplars[i]].append(i)
    else:
        groups[exemplars[i]] = [i]

print(sim)
print(resp)
print(avail)
print(crit)
print(exemplars)
print('group indeces: {0}'.format(groups))