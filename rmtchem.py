#!/usr/bin/env python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.integrate import ode
import timeit
import argparse
import sys
from progressbar import *
from scipy.optimize import root

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

def rates(X,eta,nu,k):
    return k*np.product(X**nu,axis=1)

def func(t, X, eta, nu, k, XD1, XD2):
    return XD1-XD2*X+np.sum((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis],axis=0)

def jac(X,eta,nu,k,XD1,XD2):
    return -np.diag(XD2)+np.tensordot(np.transpose((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis]),nu/X,axes=1)

def steady(X0, eta, nu, k, XD1, XD2):
    return root(lambda x:func(0,x,eta,nu,k,XD1,XD2),x0=X0,jac=lambda x:jac(x,eta,nu,k,XD1,XD2))

def integrate(X0, eta, nu, k, XD1, XD2, t1, dt, prog=False):
    n=len(X0)
    success=1
    if prog:
        pbar=ProgressBar(widgets=['Integration: ', Percentage(),Bar(), ' ', ETA()], maxval=t1)
        pbar.start()
    Xs=np.zeros((int(t1/dt),n))
    rode=ode(func).set_integrator('lsoda',rtol=1e-3,atol=1e-3,max_step=dt)
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

def quasistatic (X0, eta, nu, k, XD1s, XD2s):
    n=len(X0)
    steps=len(XD1s)
    ret=np.ones((steps,n))
    ret[0]=X0
    for m in range(steps):
        count=0
        sol=steady(ret[m],eta,nu,k,XD1s[m],XD2s[m])
        if sol.success:
            ret[m]=sol.x
        else:
            success=1
            while (not sol.success) and (count<100) and (success>0):
                X1,success=integrate(ret[m],eta,nu,k,XD1s[m],XD2s[m],5000,100,prog=False)
                ret[m]=X1[-1]
                sol=steady(ret[m],eta,nu,k,XD1s[m],XD2s[m])
                count=count+1
            if success>0 and sol.success:
                ret[m]=sol.x
            else:
                print('\n failed to integrate ')
                return ret, 0
        if m<steps-1:
            ret[m+1]=ret[m]
    return ret, 1

def hysteresis (X0, eta, nu, k, XD1s, XD2s):
    n=len(X0)
    steps=len(XD1s)
    evals1=np.zeros((steps,n),dtype=np.complex128)
    evals2=np.zeros((steps,n),dtype=np.complex128)
    Xs1,success=quasistatic(X0, eta, nu, k, XD1s, XD2s)
    Xs2=np.flip(Xs1,axis=0)
    if success>0:
        evals1=np.array([np.linalg.eig(jac(Xs1[m],eta,nu,k,XD1s[m], XD2s[m]))[0] for m in range(steps)])
        XD3s=np.flip(XD1s,axis=0)
        XD4s=np.flip(XD2s,axis=0)
        Xs2,success=quasistatic(Xs1[-1], eta, nu, k, XD3s, XD4s)
        if success>0:
            evals2=np.array([np.linalg.eig(jac(Xs2[m],eta,nu,k,XD1s[m], XD2s[m]))[0] for m in range(steps)])
            return Xs1, np.flip(Xs2,axis=0), evals1, np.flip(evals2,axis=0)

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
    parser.add_argument("--steps", type=int, default=1000, dest='steps', help='Steps for driving')
    parser.add_argument("--output", type=int, default=0, dest='output', help='1 for matrix output, 0 for none')
    args = parser.parse_args()
    n=args.n
    nr=args.nr
    nd=args.nd
    filebase=args.filebase
    output=args.output
    seed=args.seed
    steps=args.steps
    d0=args.d0
    d1min=1
    d1max=args.dmax

    np.random.seed(seed)
    d1s=np.arange(d1min,d1max,(d1max-d1min)/steps)

    start=timeit.default_timer()
    eta,nu,k,G=get_network(n,nr)
    X0=np.exp(-G)
    inds=np.argsort(np.exp(-G))[:nd]
    scales=np.exp(-G[inds])

    XD1s=np.zeros((steps,n))
    XD2s=np.zeros((steps,n))
    for m in range(steps):
        XD1s[m,inds]=d1s[m]*d0*scales
        XD2s[m,inds]=d0

    Xs1,Xs2,evals1,evals2=hysteresis(X0, eta, nu, k, XD1s, XD2s)
    mevals1=np.array([np.max(np.real(evals1[m])[np.where(np.real(evals1[m])!=0)]) for m in range(steps)])
    mevals2=np.array([np.max(np.real(evals2[m])[np.where(np.real(evals2[m])!=0)]) for m in range(steps)])
    stop=timeit.default_timer()
    print('%.3f\t%i\t%.3e\t%.3e'%(stop-start, seed, np.max(mevals1-mevals2), np.max(mevals1)))
    if output or (np.max(np.abs(mevals1-mevals2))>1e-2 or np.max(mevals1)>0) :
        np.save(filebase+'eta.npy',eta)
        np.save(filebase+'nu.npy',nu)
        np.save(filebase+'k.npy',k)
        np.save(filebase+'G.npy',G)
        np.save(filebase+'XD1s.npy',XD1s)
        np.save(filebase+'XD2s.npy',XD2s)
        np.save(filebase+'Xs1.npy',Xs1)
        np.save(filebase+'Xs2.npy',Xs2)
        np.save(filebase+'evals1.npy',evals1)
        np.save(filebase+'evals2.npy',evals2)
