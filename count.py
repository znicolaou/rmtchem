#!/usr/bin/env python
import numpy as np
c=2
bins=501
bmax=10

for c in range(2,3):
    counts=np.zeros([bins,bins])
    tot=0
    for seed in range(1024):
        try:
            vals=np.load("data/"+str(c)+"/"+str(seed)+"evals.npy")
            counts=counts+np.histogram2d(np.real(vals),np.imag(vals),bins=bins,range=[[-bmax,bmax],[-bmax,bmax]])[0]
            tot+=len(vals)
        except:
            pass
    np.save("data/"+str(c)+".npy",counts)
    print(tot, " eigenvalues for c=", c)
