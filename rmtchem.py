#!/usr/bin/env python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.integrate import ode
import timeit
import argparse
import sys
from progressbar import *
from scipy.optimize import root
import networkx as nx

def get_network(n,nr):
    eta=np.zeros((2*nr,n))
    nu=np.zeros((2*nr,n))
    k=np.zeros(2*nr)
    G=np.random.normal(loc=0, scale=1, size=n)

    for i in range(nr):
        reactants=np.random.choice(np.arange(n),size=np.random.randint(1,4),replace=False)
        products=np.random.choice(np.setdiff1d(np.arange(n),reactants),size=np.random.randint(1,4),replace=False)
        #forward
        eta[2*i,reactants]=np.random.randint(1,3,size=len(reactants))
        nu[2*i,products]=np.random.randint(1,3,size=len(products))
        k[2*i]=np.random.random()
        deltaG=np.sum(nu[2*i,products]*G[products])-np.sum(eta[2*i,reactants]*G[reactants])
        K=np.exp(-deltaG)
        #reverse
        nu[2*i+1,reactants]=eta[2*i,reactants]
        eta[2*i+1,products]=nu[2*i,products]
        k[2*i+1]=k[2*i]*K
    return eta,nu,k,G

def get_drive(eta,nu,k,G,d1min,d1max,steps):
    d1s=np.arange(d1min,d1max,(d1max-d1min)/steps)
    n=len(G)
    inds=np.random.choice(np.arange(n),size=nd,replace=False)
    scales=np.exp(-G[inds])

    XD1s=np.zeros((steps,n))
    XD2s=np.zeros((steps,n))
    for m in range(steps):
        XD1s[m,inds]=d1s[m]*d0*scales
        XD2s[m,inds]=d0

    etatot=np.zeros(n,dtype=int)
    nutot=np.zeros(n,dtype=int)
    for rind in range(nr):
        if (k[2*rind]>k[2*rind+1]):
            etatot=etatot+eta[2*rind]
            nutot=nutot+nu[2*rind]
        else:
            etatot=etatot+eta[2*rind]
            nutot=nutot+nu[2*rind]

    return XD1s,XD2s,np.sum(nutot[inds]),np.sum(etatot[inds]), np.sum(G[inds])/np.sum(G)

def rates(X,eta,nu,k):
    return k*np.product(X**nu,axis=1)

def func(t, X, eta, nu, k, XD1, XD2):
    return XD1-XD2*X+np.sum((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis],axis=0)

def jac(X,eta,nu,k,XD1,XD2):
    return -np.diag(XD2)+np.tensordot(np.transpose((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis]),nu/X,axes=1)

def steady(X0, eta, nu, k, XD1, XD2):
    return root(lambda x:func(0,x,eta,nu,k,XD1,XD2),x0=X0,jac=lambda x:jac(x,eta,nu,k,XD1,XD2), method='hybr', options={'xtol':1e-6,'diag':X0})

def integrate(X0, eta, nu, k, XD1, XD2, t1, dt, prog=False):
    n=len(X0)
    success=1
    if prog:
        pbar=ProgressBar(widgets=['Integration: ', Percentage(),Bar(), ' ', ETA()], maxval=t1)
        pbar.start()
    Xs=np.zeros((int(t1/dt),n))
    rode=ode(func).set_integrator('lsoda',rtol=1e-6,atol=1e-12,max_step=dt)
    rode.set_initial_value(X0, 0)
    rode.set_f_params(eta, nu, k, XD1, XD2)
    for n in range(int(t1/dt)):
        t=n*dt
        if prog:
            pbar.update(t)
        X=rode.integrate(rode.t + dt)
        if rode.get_return_code() < 0:
            success=0
            break
        Xs[n] = X
    if prog:
        pbar.finish()
    return Xs,success

#continue solution until a bifurcation. Return solutions and final step.
def quasistatic (X0, eta, nu, k, XD1s, XD2s):
    n=len(X0)
    steps=len(XD1s)
    sols=np.zeros((steps,n))
    evals=np.zeros((steps,n),dtype=np.complex128)
    prog=False
    if output:
        prog=True

    for m in range(steps):
        if output:
            print(m,end='\t\r')
        sol=steady(X0,eta,nu,k,XD1s[m],XD2s[m])

        if sol.success:
            sols[m]=sol.x
            evals[m]=np.linalg.eig(jac(sols[m],eta,nu,k,XD1s[m], XD2s[m]))[0]
            if np.max(np.real(evals[m]))>0:
                if np.abs(np.imag(evals[m,np.argmax(np.real(evals[m]))]))>0:
                    if output:
                        print('hopf bifurcation!',m)
                    return sols[:m+1],evals[:m+1],1
                else:
                    if output:
                        print('saddle-node bifurcation!',m)
                    return sols[:m+1],evals[:m+1],2
        else:
            if output:
                print('saddle-node bifurcation! ',m)
            return sols[:m],evals[:m],2

        #estimate new solution from jacobian
        dX=-np.linalg.solve(jac(sols[m],eta, nu, k, XD1s[m], XD2s[m]),np.diff(XD1s,axis=0)[0])
        if output and np.min(sols[m]+dX)<0:
            print('step size large (negative X0)',m)
            dX=0
        X0=sols[m]+dX
        #We could adapt step size in principle...
        if output and np.max(np.abs(dX/sols[m])) > 1e-1:
            print("step size large",m,np.max(np.abs((X0-sols[m])/X0)), flush=True)
    return sols,evals,0

if __name__ == "__main__":
    #Command line arguments
    parser = argparse.ArgumentParser(description='Noisy pendula.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output')
    parser.add_argument("--n", type=int, default=10, dest='n', help='Number of species')
    parser.add_argument("--nr", type=int, default=20, dest='nr', help='Number of reactions')
    parser.add_argument("--nd", type=int, default=1, dest='nd', help='Number of drives')
    parser.add_argument("--dmax", type=float, default=100, dest='dmax', help='Maximum drive')
    parser.add_argument("--d0", type=float, default=1e3, dest='d0', help='Drive timescale')
    parser.add_argument("--seed", type=int, default=1, dest='seed', help='Random seed')
    parser.add_argument("--steps", type=int, default=5000, dest='steps', help='Steps for driving')
    parser.add_argument("--skip", type=int, default=10, dest='skip', help='Steps to skip for output')
    parser.add_argument("--output", type=int, default=0, dest='output', help='1 for matrix output, 0 for none')
    parser.add_argument("--quasistatic", type=int, default=1, dest='quasi', help='1 for quasistatic')
    args = parser.parse_args()
    n=args.n
    nr=args.nr
    nd=args.nd
    filebase=args.filebase
    output=args.output
    quasi=args.quasi
    seed=args.seed
    steps=args.steps
    skip=args.skip
    d0=args.d0
    d1min=1
    d1max=args.dmax
    np.random.seed(seed)

    #We should find the lcc of the network and discard the rest.
    start=timeit.default_timer()
    eta,nu,k,G=get_network(n,nr)

    adj=np.zeros((n,n))
    for r in range(2*nr):
        reac=np.where(eta[r]>0)[0]
        prod=np.where(nu[r]>0)[0]
        for i in reac:
            for j in prod:
                adj[i,j]=1
            #if species are both reactants, they affect rates of change of each other
            for j in reac:
                adj[i,j]=1
    g=nx.convert_matrix.from_numpy_matrix(adj)
    lcc=np.array(list(max(nx.connected_components(g), key=len)))
    n=len(lcc)
    eta=eta[:,lcc]
    nu=nu[:,lcc]
    G=G[lcc]

    X0=np.exp(-G)

    s1=0
    s2=0
    evals,evecs=np.linalg.eig(jac(X0,eta, nu, k, np.zeros(n), np.zeros(n)))
    if np.min(np.abs(evals))<1e-8:
        s1=1
    evals,evecs=np.linalg.eig(adj[lcc][:,lcc])
    if np.min(np.abs(evals))<1e-8:
        s2=1

    XD1s,XD2s,nreac,nprod,dG=get_drive(eta,nu,k,G,d1min,d1max,steps)
    bif=-1
    Xs=np.array([])
    evals=np.array([])

    if quasi:
        Xs,evals,bif=quasistatic(X0, eta, nu, k, XD1s, XD2s)

    stop=timeit.default_timer()
    print('%.3f\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%f'%(stop-start, seed, n, s1, s2, bif, nreac, nprod, dG), flush=True)
    file=open(filebase+'out.dat','w')
    print(n,nr,nd,seed,steps,skip,d0,d1max, file=file)
    print('%.3f\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%f'%(stop-start, seed, n, s1, s2, bif, nreac, nprod, dG), file=file)
    file.close()

    if output:
        np.save(filebase+'Xs.npy',Xs[::skip])
        np.save(filebase+'evals.npy',evals[::skip])
