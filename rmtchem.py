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
        sol=steady(X0,eta,nu,k,XD1s[m],XD2s[m])

        if sol.success:
            sols[m]=sol.x
            evals[m]=np.linalg.eig(jac(sols[m],eta,nu,k,XD1s[m], XD2s[m]))[0]
            if(np.max(np.real(evals[m]))>0):
                if output:
                    print('hopf bifurcation!',m)
                return sols[:m],evals[:m],1
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


def hysteresis (X0, eta, nu, k, XD1s, XD2s):
    n=len(X0)
    steps=len(XD1s)
    evals1=np.zeros((steps,n),dtype=np.complex128)
    evals2=np.zeros((steps,n),dtype=np.complex128)
    if output:
        print('forward', flush=True)
    Xs1,evals1=quasistatic(X0, eta, nu, k, XD1s, XD2s)
    evals2=np.flip(evals1.copy())
    mmax=len(evals1)
    XD3s=np.flip(XD1s[:mmax],axis=0)
    XD4s=np.flip(XD2s[:mmax],axis=0)
    if output:
        print('reverse', flush=True)
    Xs2,evals3=quasistatic(Xs1[mmax-1], eta, nu, k, XD3s, XD4s)
    mmin=len(evals3)
    evals2[:mmin]=evals3
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
    parser.add_argument("--steps", type=int, default=5000, dest='steps', help='Steps for driving')
    parser.add_argument("--skip", type=int, default=10, dest='skip', help='Steps to skip for output')
    parser.add_argument("--output", type=int, default=0, dest='output', help='1 for matrix output, 0 for none')
    args = parser.parse_args()
    n=args.n
    nr=args.nr
    nd=args.nd
    filebase=args.filebase
    output=args.output
    seed=args.seed
    steps=args.steps
    skip=args.skip
    d0=args.d0
    d1min=1
    d1max=args.dmax

    np.random.seed(seed)
    d1s=np.arange(d1min,d1max,(d1max-d1min)/steps)

    start=timeit.default_timer()
    eta,nu,k,G=get_network(n,nr)
    if np.min(np.max(eta,axis=0)+np.max(nu,axis=0)) < 1:
        stop=timeit.default_timer()
        print('%.3f\t%i\t%i\t%.3e\t%.3e'%(stop-start, seed, 0, -1.0, -1.0), flush=True)
        quit()
    X0=np.exp(-G)
    inds=np.argsort(np.exp(-G))[:nd]
    scales=np.exp(-G[inds])

    XD1s=np.zeros((steps,n))
    XD2s=np.zeros((steps,n))
    for m in range(steps):
        XD1s[m,inds]=d1s[m]*d0*scales
        XD2s[m,inds]=d0
    Xs1,evals1,bif=quasistatic(X0, eta, nu, k, XD1s, XD2s)

    stop=timeit.default_timer()
    print('%.3f\t%i\t%i'%(stop-start, seed, bif), flush=True)
    file=open(filebase+'out.dat','w')
    print(n,nr,nd,seed,steps,skip,d0,d1max, file=file)
    print('%.3f\t%i\t%i'%(stop-start, seed, bif), file=file)
    file.close()

    if output or (bif != 0) :
        np.save(filebase+'Xs.npy',Xs1[::skip])
        np.save(filebase+'evals.npy',evals1[::skip])


    # mmax=len(evals1)
    # Xs1,Xs2,evals1,evals2=hysteresis(X0, eta, nu, k, XD1s, XD2s)
    # mevals1=np.array([np.max(np.real(evals1[m])) for m in range(mmax)])
    # mevals2=np.array([np.max(np.real(evals2[m])) for m in range(mmax)])
    # print('%.3f\t%i\t%i\t%.3e\t%.3e'%(stop-start, seed, mmax, np.max(mevals1-mevals2), np.max(mevals1)), flush=True)
    # if output or (np.max(np.abs(mevals1-mevals2))>1e-2 or np.max(mevals1)>0) :

    # count=0
    # success=1
    # X0=X0*(1+(np.random.random(size=n)-0.5)*1e-2) #perturb ic
    # while (not sol.success) and (count<10) and (success>0) and (np.min(X0)>0):
    #     X1,success=integrate(X0,eta,nu,k,XD1s[m],XD2s[m],1000,0.1,prog=prog)
    #     X0=X1[-1]
    #     sol=steady(X0,eta,nu,k,XD1s[m],XD2s[m])
    #     count=count+1
    #     if output:
    #         print(count,sol.message, success, flush=True)
    # if success>0 and sol.success and np.min(sol.x)>0:
    #     if output:
    #         print('new branch found')
    #     sols[m]=sol.x
    #     evals[m]=np.linalg.eig(jac(sols[m],eta,nu,k,XD1s[m], XD2s[m]))[0]
    #     return sols[:m+1],evals[:m+1]
    # else:
    #     if output:
    #         print('failed - no fixed points?')
    #     return sols[:m],evals[:m]
