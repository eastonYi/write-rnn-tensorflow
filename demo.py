import numpy as np
import time
import os
import pickle
import svgwrite

from IPython.display import SVG, display
import matplotlib.pyplot as plt
from utils import DataLoader

data_loader = DataLoader(10, 20, 1)
data_loader.reset_batch_pointer()

pos_x =data_loader.raw_data[0][:,0]
pos_y =data_loader.raw_data[0][:,1]
m =0; n=0
a =[];b =[]
for x, y in zip(pos_x, pos_y):
    m +=x
    n +=y
    a.append(m)
    b.append(n)

# plt.plot(a, b)
# plt.show()

with open('strokes.pkl', 'rb') as f:
    m, n=pickle.load(f)
pos_x =m[:,0]
pos_y =m[:,1]
m =0; n=0
a =[];b =[]
for x, y in zip(pos_x, pos_y):
    m +=x
    n +=y
    a.append(m)
    b.append(n)
plt.plot(a, b)
plt.show()
