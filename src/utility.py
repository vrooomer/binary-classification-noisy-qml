import json
import numpy as np
import random
import math

# ================= GRID FUNCTIONS

def save_grid(grid_dict, filename):
    # Make the grid serializable
    grid_dict['grid'] = grid_dict['grid'].tolist()
    # Make the unitary serializable
    tmp = []
    for line in grid_dict['unitary'].tolist():
        tmp.append([str(c) for c in line])
    grid_dict['unitary'] = tmp
    with open(filename, 'w') as file:
        json.dump(grid_dict, file)

def load_grid(grid_filename):
    # TODO: convert back to narray (complex if needed)
    grid_dict = {}
    with open(grid_filename, 'r') as file:
        grid_dict = json.load(file)
    return grid_dict


# ================= DATASET FUNCTIONS

"""
Creates a dataset of the given size, using the specified grid. Only creates points with +/- labels. 
If balanced=True, we make sure to create 50/50 datapoints for each label. 
"""
def create_dataset(grid_dict, dataset_size, balanced=True, shuffled=True):
    dataset = []
    positive_size = math.floor(dataset_size/2) # For balanced dataset
    grid_size = grid_dict['grid_size']
    while len(dataset) < dataset_size:
        r1 = random.random()
        r2 = random.random()
        label = grid_dict['grid'][math.floor(r2*grid_size)][math.floor(r1*grid_size)]
        data_point = {'input': [2*math.pi*r1, 2*math.pi*r2], 'label': label}
        enough_positive = len(dataset) >= positive_size # For balanced dataset
        if label != 0:
            if not balanced:
                dataset.append(data_point)
            else: # For balanced dataset
                if (label == 1 and not enough_positive) or (label == -1 and enough_positive):
                    dataset.append(data_point)
    if shuffled:
        random.shuffle(dataset)
    return dataset

# ================= GENERAL FUNCTIONS

def save_list(listdata, filename):
    with open(filename, 'w') as file:
        for d in listdata[:-1]:
            json.dump(d, file)
            file.write('\n')
        json.dump(listdata[-1], file) # To avoid writing empty line

def load_list(filename):
    listdata = []
    with open(filename, 'r') as file:
        for line in file:
            listdata.append(json.loads(line))
    return listdata

# ================= LATEX FUNCTIONS

"""
Converts a 2d list of complex number to a latex matrix (multiplied by 10 and rounded to ints)
"""
def list_to_latex_matrix(mat_list):
    text = r"\begin{pmatrix} "
    for row in mat_list:
        for el in row:
            z = complex(el)
            rz = round(z.real*10)
            iz = round(z.imag*10)
            sign = '+' if iz > 0 else ''
            text += f"{rz}{sign}{iz}i &"
        text += r'\\'
    text += r"\end{pmatrix}"
    return text
