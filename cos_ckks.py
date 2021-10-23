# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 21:04:50 2020

@author: zhangtianxia
"""


import numpy as np
import tenseal as ts
import time

from math import ceil
# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40



net1 = np.load('net1.npy')
net2 = np.load('net2.npy')
net3 = np.load('net3.npy')
net4 = net1+net2+net3

width = 3000
scale= 100
b_len =  ceil(len(net1)/width)*width-len(net1)
net1 = np.pad(net1,(0,b_len)).reshape(-1,width)
net2 = np.pad(net2,(0,b_len)).reshape(-1,width)
net3 = np.pad(net3,(0,b_len)).reshape(-1,width)



enet1 = []
enet2 = []
enet3 = []
enet = []
dnet = []
fenzi = 0
fenmu1=0
fenmu2=0
# fenzi = []
e = len(net1)
# encrypt model parameters  
for i in range(e):
    enet1.append(ts.ckks_vector(context, net1[i])*scale)
    enet2.append(ts.ckks_vector(context, net2[i])*scale)
    enet3.append(ts.ckks_vector(context, net3[i])*scale)
    
    enet.append(ts.ckks_vector(context,net1[i])+ts.ckks_vector(context,net2[i])+
                ts.ckks_vector(context,net3[i]))  # model parameter aggregation 


    
    
# calcualte cosine similarity in ciphertext ()   
for i in range(e):
    fenzi +=np.array((enet1[i].dot(enet2[i])).decrypt())
    fenmu1 +=np.array((enet1[i].dot(enet1[i])).decrypt())
    fenmu2 +=np.array((enet2[i].dot(enet2[i])).decrypt())   

fenmu = np.sqrt(fenmu1)*np.sqrt(fenmu2)   
    
cos = fenzi/fenmu 

print(cos)
    
    
    
    
