#!/usr/bin/env python
import numpy as np
c=2
bins=101
bmax=6

n=100
for c in range(2,3):
    counts=np.zeros([bins,bins])
    avg=np.zeros([50,50,2,2])
    tot=0
    for seed in range(1,8000):
        try:
            vals=np.load("data/"+str(c)+"_"+str(n)+"/"+str(seed)+"evals.npy")
            counts=counts+np.histogram2d(np.real(vals),np.imag(vals),bins=bins,range=[[-bmax,bmax],[-bmax,bmax]])[0]
            tot+=len(vals)
        except:
            print("data/"+str(c)+"_"+str(n)+"/"+str(seed)+"evals.npy"+" not found")
            pass
        try:
            vals2=np.load("data/"+str(c)+"_"+str(n)+"/"+str(seed)+"g.npy")
            avg=avg+vals
        except:
            print("data/"+str(c)+"_"+str(n)+"/"+str(seed)+"g.npy"+" not found")
            pass
    np.save("data/"+str(c)+"_"+str(n)+".npy",counts)
    np.save("data/"+str(c)+"_"+str(n)+"g.npy",avg)
    print(tot, " eigenvalues for c=", c)
