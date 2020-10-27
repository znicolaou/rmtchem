#!/usr/bin/env python
import numpy as np
c=2
bins=101
bmax=6

n=50
for c in range(2,3):
    counts=np.zeros([bins,bins])
    avg=np.zeros([25,2,2])
    tot=0
    for seed in range(1,1024):
        try:
            vals=np.load("data/"+str(c)+"_"+str(n)+"/"+str(seed)+"evals.npy")
            counts=counts+np.histogram2d(np.real(vals),np.imag(vals),bins=bins,range=[[-bmax,bmax],[-bmax,bmax]])[0]
            tot+=len(vals)
            g1=np.load("data/"+str(c)+"_"+str(n)+"/"+str(seed)+"g.npy")

            vals2=np.diagonal(np.load("data/"+str(c)+"_"+str(n)+"/"+str(seed)+"g.npy"),axis1=3,axis2=4)
            avg=avg+np.mean(vals2,axis=(3))/1024

        except:
            print("seed "+str(seed)+" not found")
            pass
    np.save("data/"+str(c)+"_"+str(n)+".npy",counts)
    np.save("data/"+str(c)+"_"+str(n)+"g.npy",avg)
    print(tot, " eigenvalues for c=", c)
