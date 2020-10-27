#!/usr/bin/env python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.integrate import ode
import timeit
import argparse
import sys
from progressbar import *


def regular (n, c, mu, sigma):
    rows=np.array([])
    cols=np.array([])
    dat=np.array([])
    in_degs=np.zeros(n)
    for i in range(n):
        for edge in range(c):
            potential=np.setdiff1d(np.where(in_degs<c)[0],[i])
            potential=np.setdiff1d(potential,rows[np.where(cols==i)[0]])
            potential=np.setdiff1d(potential,cols[np.where(rows==i)[0]])
            potential=np.setdiff1d(potential,rows[np.where(cols==i)[0]])
            if(len(potential)==0):
                print("failed, trying again")
                return regular(c, n, mu, sigma)
            j=np.random.choice(potential)
            dat=np.append(dat,mu+np.random.normal(loc=mu,scale=sigma))
            rows=np.append(rows,i)
            cols=np.append(cols,j)
            in_degs[j]+=1

    A=csr_matrix((dat, (rows, cols)), shape=(n, n)).toarray()
    return A
def erdos (n, c, mu, sigma):
    rows=[]
    cols=[]
    dat=[]
    while (len(rows)<c*n):
        row=np.random.randint(n)
        col=np.random.randint(n)

        exclude=np.append(np.array(cols)[np.where(np.array(rows)==row)[0]], np.array(rows)[np.where(np.array(cols)==row)[0]])

        if not np.any((row==np.array(rows))*(col==np.array(cols))):
            if not np.any((col==np.array(rows))*(row==np.array(cols))):
                if not np.any(col==exclude):
                    rows.append(row)
                    cols.append(col)
                    dat.append(np.random.normal(loc=mu,scale=sigma))

    A=csr_matrix((dat, (rows, cols)), shape=(n, n)).toarray()
    return A
def B(A, z,eta):
    n=len(A)
    return np.block([[eta*np.diag(np.ones(n)), -1j*(A-z*np.diag(np.ones(n)))], [-1j*np.transpose(np.conjugate((A-z*np.diag(np.ones(n))))),eta*np.diag(np.ones(n))]])
def G(A, z, eta):
    n=len(A)
    return -1j*np.linalg.inv(B(A, z,eta))[n:,:n]
def rho(A, z, eta, epsilon):
    n=len(A)
    return 1j/(n*np.pi)*np.trace(G(A,z,eta)-G(A,z-epsilon,eta)-1j*G(A,z,eta)+1j*G(A,z-1j*epsilon,eta))/(2*epsilon)

def g(A,z,eta):
    n=len(A)
    binv=np.linalg.inv(B(A,z,eta))
    ret=1j*np.array([[binv[:n,:n],binv[:n,n:]],[binv[n:,:n],binv[n:,n:]]])
    return ret

def gr(A,z,eta):
    n=len(A)
    ret=np.zeros((2,2,n,n,n),dtype=np.complex128)
    b=B(A, z,eta)
    j2=np.arange(n-1)
    k2=np.arange(n-1)[:,np.newaxis]

    for l in range(n):
        indices=np.setdiff1d(np.arange(n),[l])
        indices=np.append(indices,indices+n)
        binv=np.linalg.inv(b[indices[:,np.newaxis],indices])
        j=np.setdiff1d(np.arange(n),[l])
        k=np.setdiff1d(np.arange(n),[l])[:,np.newaxis]
        ret[:,:,j,k,l] = 1j*np.array([[binv[j2,k2],binv[j2,k2+n-1]],[binv[j2+n-1,k2],binv[j2+n-1,k2+n-1]]])
    return ret

if __name__ == "__main__":
    #Command line arguments
    parser = argparse.ArgumentParser(description='Noisy pendula.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output')
    parser.add_argument("--n", type=int, default=10, dest='n', help='Number of nodes')
    parser.add_argument("--c", type=int, default=2, dest='c', help='Connectivity')
    parser.add_argument("--mu", type=float, default=1.0, dest='mu', help='Mean of entries')
    parser.add_argument("--type", type=int, default=1, dest='type', help='1 for regular, 2 for Erdos-Renyi')
    parser.add_argument("--sigma", type=float, default=1.0, dest='sigma', help='Standard deviation of entries')
    parser.add_argument("--output", type=int, default=1, dest='output', help='1 for matrix output, 0 for none')
    parser.add_argument("--seed", type=int, default=1, dest='seed', help='Random seed for the network')
    parser.add_argument("--zr0", type=float, default=0, dest='zr0', help='Initial Re(z) for generalized resolvant')
    parser.add_argument("--zi0", type=float, default=0, dest='zi0', help='Initial Im(z) for generalized resolvant')
    parser.add_argument("--zr1", type=float, default=0, dest='zr1', help='Final Re(z) for generalized resolvant')
    parser.add_argument("--zi1", type=float, default=10, dest='zi1', help='Final Im(z) for generalized resolvant')
    parser.add_argument("--gnum", type=int, default=25, dest='gnum', help='Number of g to evaluate')
    parser.add_argument("--eta", type=float, default=1e-2, dest='eta', help='Regularization parameter')
    args = parser.parse_args()
    n=args.n
    c=args.c
    mu=args.mu
    sigma=args.sigma
    filebase=args.filebase
    output=args.output
    seed=args.seed
    type=args.type
    z0=args.zr0+1j*args.zi0
    z1=args.zr1+1j*args.zr1
    gnum=args.gnum
    eta=args.eta
    np.random.seed(seed)

    start=timeit.default_timer()
    if type==1:
        A=regular(c,n,mu,sigma)
    else:
        A=erdos(c,n,mu,sigma)
    stop=timeit.default_timer()
    print("Generated random regular ", n, "x", n, " matrix with connectivity ", c , " in ", stop-start, "seconds")
    sys.stdout.flush()

    start=timeit.default_timer()
    evals,evecs=np.linalg.eig(A)
    stop=timeit.default_timer()
    if output == 1:
        np.save(filebase+"mat.npy",A)
    np.save(filebase+"evals.npy",evals)

    print("Calculated eigenvalues in ", stop-start, "seconds")

    glst=np.zeros((gnum,2,2,n,n),dtype=np.complex128)
    grlst=np.zeros((gnum,2,2,n,n,n),dtype=np.complex128)
    Zlst=np.zeros((gnum,2,2),dtype=np.complex128)
    for i in range(gnum):
        z=z0+(z1-z0)/gnum*i
        Z=np.array([[0,z],[z.conjugate(),0]])
        Zlst[i]=Z-1j*eta*np.array([[1,0],[0,1]])
        glst[i]=g(A,z,eta)
        grlst[i]=gr(A,z,eta)

    np.save(filebase+"z.npy",Zlst)
    np.save(filebase+"g.npy",glst)
    np.save(filebase+"gr.npy",grlst)
    stop=timeit.default_timer()
    print("Calculated "+str(gnum)+" generalized resolvants in", stop-start, "seconds")
