#!/usr/bin/env python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.integrate import ode
import timeit
import argparse
import sys
from progressbar import *
from scipy.optimize import fsolve


def rates(X,eta,nu,k):
    return k*np.product(X**nu,axis=1)

def func(t, X, eta, nu, k):
    return np.sum((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis],axis=0)

def jac(X,eta,nu,k):
    return np.tensordot(np.transpose((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis]),nu/X,axes=1)

def steady(n, t1, dt, eta, nu, k):
    pbar=ProgressBar(widgets=['Integration: ', Percentage(),Bar(), ' ', ETA()], maxval=t1)
    pbar.start()
    X0 = np.random.random(n)
    rode=ode(func).set_integrator('lsoda',rtol=1e-3,atol=1e-3,max_step=dt)
    rode.set_initial_value(X0, 0)
    rode.set_f_params(eta, nu, k)
    for n in range(int(t1/dt)):
        t=n*dt
        pbar.update(t)
        X=rode.integrate(rode.t + dt)
        Xs[n] = X
    pbar.finish()

def func2(X, eta, nu, k, XD):
    return XD+np.sum((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis],axis=0)

def func3(t, X, eta, nu, k, XD):
    return XD+np.sum((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis],axis=0)

def jac2(X,eta,nu,k,XD):
    return np.tensordot(np.transpose((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis]),nu/X,axes=1)

def steady1(X0, eta, nu, k, XD):
    return fsolve(func2,x0=X0,args=(eta,nu,k,XD),fprime=jac2)

def steady2(n, t1, dt, eta, nu, k, XD):
    pbar=ProgressBar(widgets=['Integration: ', Percentage(),Bar(), ' ', ETA()], maxval=t1)
    pbar.start()
    X0 = np.random.random(n)
    XX=np.zeros((int(t1/dt),n))
    rode=ode(func3).set_integrator('lsoda',rtol=1e-3,atol=1e-3,max_step=dt)
    rode.set_initial_value(X0, 0)
    rode.set_f_params(eta, nu, k, XD)
    for n in range(int(t1/dt)):
        t=n*dt
        pbar.update(t)
        X=rode.integrate(rode.t + dt)
        XX[n] = X
    pbar.finish()
    return XX

if __name__ == "__main__":
    #Command line arguments
    parser = argparse.ArgumentParser(description='Noisy pendula.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output')
    parser.add_argument("--n", type=int, default=10, dest='n', help='Number of species')
    parser.add_argument("--nr", type=int, default=5, dest='nr', help='Number of reactions')
    parser.add_argument("--c", type=int, default=2, dest='c', help='Connectivity')
    parser.add_argument("--mu", type=float, default=1.0, dest='mu', help='Mean of entries')
    parser.add_argument("--tmax", type=float, default=10, dest='t1', help='Max time')
    parser.add_argument("--dt", type=float, default=0.1, dest='dt', help='Time step')
    parser.add_argument("--reversible", type=float, default=1., dest='reversible', help='Reversibility')
    parser.add_argument("--type", type=int, default=1, dest='type', help='1 for regular, 2 for chemistry')
    parser.add_argument("--sigma", type=float, default=1.0, dest='sigma', help='Standard deviation of entries')
    parser.add_argument("--output", type=int, default=1, dest='output', help='1 for matrix output, 0 for none')
    parser.add_argument("--iseed", type=int, default=1, dest='iseed', help='Random seed')
    parser.add_argument("--nseed", type=int, default=1, dest='nseed', help='Random seed for the network')
    args = parser.parse_args()
    n=args.n
    c=args.c
    mu=args.mu
    sigma=args.sigma
    filebase=args.filebase
    output=args.output
    iseed=args.iseed
    nseed=args.nseed
    nr=args.nr
    type=args.type
    t1=args.t1
    dt=args.dt
    G=np.random.random(n)
    reversible=args.reversible
    np.random.seed(nseed)

    start=timeit.default_timer()
    eta=np.zeros((2*nr,n))
    nu=np.zeros((2*nr,n))
    k=np.zeros(2*nr)

    for i in range(nr):
        reactants=np.random.choice(np.arange(n),size=2,replace=False)
        products=np.random.choice(np.setdiff1d(np.arange(n),reactants),size=2,replace=False)
        #forward
        eta[2*i,reactants[0]]=1
        eta[2*i,reactants[1]]=1
        nu[2*i,products[0]]=1
        nu[2*i,products[1]]=1
        # k[2*i]=1
        k[2*i]=np.random.random()
        deltaG=np.sum(G[products])-np.sum(G[reactants])
        K=np.exp(-deltaG)
        print(K)
        #reverse
        nu[2*i+1,reactants[0]]=1
        nu[2*i+1,reactants[1]]=1
        eta[2*i+1,products[0]]=1
        eta[2*i+1,products[1]]=1
        k[2*i+1]=k[2*i]*K

    pbar=ProgressBar(widgets=['Integration: ', Percentage(),Bar(), ' ', ETA()], maxval=t1)
    pbar.start()
    Xs=np.zeros((int(t1/dt),n))
    steady(n,t1,dt,eta,nu,k)
    stop=timeit.default_timer()
    print("Calculated dynamics in ", stop-start, "seconds")

    A=jac(Xs[-1],eta,nu,k)
    np.save(filebase+"X",Xs)
    np.save(filebase+"nu",nu)
    np.save(filebase+"eta",eta)
    out=open(filebase+"out.dat","w")
    print('%i %i' % (n, nr), file=out)
    out.close()



    start=timeit.default_timer()
    evals,evecs=np.linalg.eig(A)
    stop=timeit.default_timer()
    if output == 1:
        np.save(filebase+"mat.npy",A)
    np.save(filebase+"evals.npy",evals)

    print("Calculated eigenvalues in ", stop-start, "seconds")
