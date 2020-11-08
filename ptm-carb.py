import numpy as np; pi = np.pi
import scipy.io as spio
import networkx as nx
from qutip import *
from ptm.ptm import PTM
from ptm.ptm import *
import scipy.io as spio

for i in range(10):
    filename = "instance_8_"+str(i+1)
    mat = spio.loadmat("instances/" + filename + ".mat")
    instance = mat['instance']
    cv_ptm = PTM()
    for p in range(1):
        gamma = instance['gamma'][0,0][0][p].flatten()
        temp = cv_ptm.carb(gamma)
        qsave(temp, "results/"+filename+"_carb_p_"+str(p+1)+"_1")
