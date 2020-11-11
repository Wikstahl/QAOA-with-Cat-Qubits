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
    gamma = instance['gamma'][0,0][0][p-1].flatten()
    cv_ptm = PTM()
    for k in range(1):
        temp = cv_ptm.carb(gamma[k])
        qsave(temp, "results/"+filename+"_carb_p_"+str(k+1)+"_2")
