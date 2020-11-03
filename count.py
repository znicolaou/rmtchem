#!/usr/bin/env python
import numpy as np
import argparse

if __name__ == "__main__":
    #Command line arguments
    parser = argparse.ArgumentParser(description='Random matrices.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output')
    parser.add_argument("--output", type=int, default=1, dest='output', help='1 for matrix output, 0 for none')
    parser.add_argument("--seed", type=int, default=1, dest='seed', help='Random seed for the network')
    parser.add_argument("--zr0", type=float, default=-2, dest='zr0', help='Initial Re(z) for generalized resolvant')
    parser.add_argument("--zi0", type=float, default=-2, dest='zi0', help='Initial Im(z) for generalized resolvant')
    parser.add_argument("--zr1", type=float, default=2, dest='zr1', help='Final Re(z) for generalized resolvant')
    parser.add_argument("--zi1", type=float, default=2, dest='zi1', help='Final Im(z) for generalized resolvant')
    parser.add_argument("--gnum", type=int, default=201, dest='gnum', help='Number of g to evaluate')
    parser.add_argument("--eta", type=float, default=1e-2, dest='eta', help='Regularization parameter')
    args = parser.parse_args()

    filebase=args.filebase
    output=args.output
    seed=args.seed
    zr0=args.zr0
    zi0=args.zi0
    zr1=args.zr1
    zi1=args.zi1
    gnum=args.gnum
    eta=args.eta

    counts=np.zeros([gnum,gnum])
    avg=np.zeros([gnum,gnum,2,2],dtype=np.complex128)
    tot=0
    for seed in range(seed):
        try:
            vals=np.load(filebase+"/"+str(seed)+"evals.npy")
            counts=counts+np.histogram2d(np.real(vals),np.imag(vals),bins=gnum,range=[[zr0,zr1],[zi0,zi1]])[0]
            tot+=len(vals)
        except:
            print(filebase+"/"+str(seed)+"evals.npy"+" not found")
            pass
        try:
            vals2=np.load(filebase+"/"+str(seed)+"g.npy")
            avg=avg+vals2
        except:
            print(filebase+"/"+str(seed)+"g.npy"+" not found")
            pass
    np.save(filebase+".npy",counts/tot)
    np.save(filebase+"g.npy",avg/tot)
    print(tot, " eigenvalues")
