import numpy as np; pi = np.pi
import scipy.io as spio
import networkx as nx
from qutip import *
from ptm.ptm import PTM
from ptm.ptm import *
import scipy.io as spio

p = 2
for i in range(10):
    filename = "instance_8_"+str(i+1)
    mat = spio.loadmat("instances/" + filename + ".mat")
    instance = mat['instance']
    cv_ptm = PTM()
    for k in range(p):
        gamma = instance['gamma'][0,0][0][k].flatten()
        temp = cv_ptm.carb(gamma)
        qsave(temp, "results/"+filename+"_carb_p_"+str(k+1)+"_2")
