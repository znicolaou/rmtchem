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

def get_network(n,nr,na=0):
    eta=np.zeros((2*nr,n))
    nu=np.zeros((2*nr,n))
    k=np.zeros(2*nr)
    G=np.random.normal(loc=0, scale=1.0, size=n)

    for i in range(nr):
        reactants=np.random.choice(np.arange(n),size=np.random.randint(1,4),replace=False)
        products=np.random.choice(np.setdiff1d(np.arange(n),reactants),size=np.random.randint(1,4),replace=False)

        eta[2*i,reactants]=np.random.randint(1,3,size=len(reactants))
        nu[2*i,products]=np.random.randint(1,3,size=len(products))

        if i<na:
            auto=np.random.choice(reactants)
            nu[2*i,auto]=eta[2*i,auto]

        nu[2*i+1]=eta[2*i]
        eta[2*i+1]=nu[2*i]

        #Randomly sample the rate constant in the deltaG>0 direction
        deltaG=np.sum(nu[2*i]*G)-np.sum(eta[2*i]*G)
        if deltaG>0:
            k[2*i]=np.random.exponential(scale=deltaG)
            k[2*i]=np.random.random()
            k[2*i+1]=k[2*i]*np.exp(-deltaG)
        else:
            k[2*i+1]=np.random.exponential(scale=-deltaG)
            k[2*i+1]=np.random.random()
            k[2*i]=k[2*i+1]*np.exp(deltaG)

    return eta,nu,k,G

def get_drive(eta,nu,k,G,d0,nd):
    n=len(G)
    # inds=np.random.choice(np.arange(n),size=nd,replace=False)
    inds=np.argsort(G)[-nd:] #drive the most stable species, to avoid large concentration ratios
    XD1=np.zeros(n)
    XD2=np.zeros(n)
    XD1[inds]=d0*np.exp(-G[inds])
    XD2[inds]=d0
    return XD1, XD2, inds

def rates(X,eta,nu,k):
    return k*np.product(X**nu,axis=1)

def func(t, X, eta, nu, k, XD1, XD2):
    return XD1-XD2*X+rates(X,eta,nu,k).dot(eta-nu)

def jac(t,X,eta,nu,k,XD1,XD2):
    return -np.diag(XD2)+np.transpose((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis]).dot(nu/X)

def getNu2(nu):
    #Could be vectorized perhaps, but only need to calculate once
    m,n=nu.shape
    nu2=np.zeros((m,n,n))
    for l in range(m):
        for i in range(n):
            nu2[l,i,i]=nu[l,i]*(nu[l,i]-1)
            for j in range(i):
                nu2[l,i,j]=nu[l,i]*nu[l,j]
                nu2[l,j,i]=nu[l,i]*nu[l,j]
    return nu2

def hess(t,X,eta,nu,k,XD1,XD2,nu2=[]):
    if len(nu2)==0:
        nu2=getNu2(nu)
    return np.tensordot((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis],nu2/X[np.newaxis,np.newaxis,:]/X[np.newaxis,:,np.newaxis],axes=([0],[0]))

def steady(X0, eta, nu, k, XD1, XD2):
    sol=root(lambda x:func(0,x,eta,nu,k,XD1,XD2),x0=X0,jac=lambda x:jac(0,x,eta,nu,k,XD1,XD2), method='hybr', tol=1e-6)
    return sol.success,sol.x

def integrate(X0, eta, nu, k, XD1, XD2, t1, dt, prog=False):
    n=len(X0)
    success=1
    if prog:
        pbar=ProgressBar(widgets=['Integration: ', Percentage(),Bar(), ' ', ETA()], maxval=t1)
        pbar.start()
    Xs=np.zeros((int(t1/dt),n))
    rode=ode(func,jac).set_integrator('lsoda',rtol=1e-6,atol=1e-12,max_step=dt/10)
    rode.set_initial_value(X0, 0)
    rode.set_f_params(eta, nu, k, XD1, XD2)
    rode.set_jac_params(eta, nu, k, XD1, XD2)
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

def snfunc(X0,epsilon0,Xs,epsilons,alpha,beta,evecs,ind):
    inds=np.setdiff1d(np.arange(len(X0)),[ind])
    S0=(Xs-X0)
    S1=((((epsilons-epsilon0)*alpha/2)**0.5)[:,np.newaxis]*evecs[:,ind])
    S2=-(epsilons-epsilon0)[:,np.newaxis]*np.sum(beta[inds]*evecs[:,inds],axis=1)
    return np.sum(np.linalg.norm(S0+S1+S2,axis=1))

#We should have epsilon passed rather than XD1s and XD2s
def quasistatic (X0, eta, nu, k, XD1, XD2, epsilon0, epsilon1, steps, output=True, stop=True):
    n=len(X0)
    nr=int(len(nu)/2)

    sols=np.zeros((steps,n))
    evals=np.zeros((steps,n),dtype=np.complex128)
    prog=False
    if output:
        prog=True

    nu2=getNu2(nu)
    drives=np.zeros(n)
    drives[np.where(XD1!=0)[0]]=1
    epsilon=epsilon0
    depsilon=(epsilon1-epsilon0)/steps
    for m in range(steps):
        if output:
            print(m,end='\t\r')
        success,solx=steady(X0,eta,nu,k,(1+epsilon)*XD1,XD2)

        if success:
            sols[m]=solx
            mat=jac(0,sols[m],eta,nu,k,(1+epsilon)*XD1,XD2)
            evals[m],evecs=np.linalg.eig(mat)
            if np.max(np.real(evals[m]))>0:
                if np.abs(np.imag(evals[m,np.argmax(np.real(evals[m]))]))>0:
                    if output:
                        print('\nhopf bifurcation!',m)
                    if stop:
                        return sols[:m+1],evals[:m+1],1
                else:
                    if output:
                        print('\nsaddle-node bifurcation (transcritical or pitchfork)!',m)
                    return sols[:m+1],evals[:m+1],2
            #estimate new solution from jacobian
            #We should do adapative steps, and add the least-squares stopping condition
            #We could switch branches in the stopping condition if we like
            epsilon=epsilon+depsilon #step size could be adaptive here
            ind=np.argmax(np.real(evals[m]))
            dX=-np.linalg.solve(mat,depsilon*XD1)
            if np.min(sols[m]+dX)<0:
                if output:
                    print('step size large (negative X0)',m,'\t\r',end='')
                dX=0
            X0=sols[m]+dX
            if output and np.max(np.abs(dX/sols[m])) > 1e-1:
                print("step size large",m,np.max(np.abs((X0-sols[m])/X0)),'\t\n',end='', flush=True)

            einv=np.linalg.inv(evecs)
            beta=XD1.dot(einv)
            alpha=hess(0,sols[m],eta,nu,k,(1+epsilon)*XD1,XD2,nu2).dot(evecs[:,ind]).dot(evecs[:,ind]).dot(einv[ind])/beta[ind]
            if(np.abs(evals[m,ind])<1e-2):
                epsilons=epsilon-depsilon*np.flip(np.arange(2))
                # print(snfunc(sols[m],epsilon,sols[m-1:m+1],epsilons,alpha,beta,evecs,ind))

        else:
            if output:
                print('\nFailed to converge! ',m)
            return sols[:m],evals[:m],-1


    return sols,evals,0

if __name__ == "__main__":
    #Command line arguments
    parser = argparse.ArgumentParser(description='Random chemical reaction networks.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output')
    parser.add_argument("--n", type=int, default=10, dest='n', help='Number of species')
    parser.add_argument("--nr", type=int, default=20, dest='nr', help='Number of reactions')
    parser.add_argument("--nd", type=int, default=1, dest='nd', help='Number of drives')
    parser.add_argument("--dmax", type=float, default=100, dest='dmax', help='Maximum drive')
    parser.add_argument("--type", type=int, default=0, dest='type', help='Type of adjacency matrix. 0 for chemical networks, 1 for ER networks.')
    parser.add_argument("--d0", type=float, default=1e3, dest='d0', help='Drive timescale')
    parser.add_argument("--seed", type=int, default=1, dest='seed', help='Random seed')
    parser.add_argument("--steps", type=int, default=5000, dest='steps', help='Steps for driving')
    parser.add_argument("--na", type=int, default=0, dest='na', help='Number of autocatalytic reactions')
    parser.add_argument("--skip", type=int, default=10, dest='skip', help='Steps to skip for output')
    parser.add_argument("--output", type=int, default=0, dest='output', help='1 for matrix output, 0 for none')
    parser.add_argument("--quasistatic", type=int, default=1, dest='quasi', help='1 for quasistatic')
    parser.add_argument("--rank", type=int, default=1, dest='rank', help='1 for rank calculation')
    args = parser.parse_args()
    n=args.n
    nr=args.nr
    nd=args.nd
    na=args.na
    filebase=args.filebase
    output=args.output
    quasi=args.quasi
    rank=args.rank
    seed=args.seed
    steps=args.steps
    skip=args.skip
    d0=args.d0
    d1min=1
    d1max=args.dmax
    np.random.seed(seed)

    #We should find the lcc of the network and discard the rest.
    start=timeit.default_timer()
    eta,nu,k,G=get_network(n,nr,na)

    if args.type==0:
        row,col=np.where(eta[::2]-nu[::2]!=0)
        data=(eta[::2]-nu[::2])[row,col]
        A=csr_matrix((data,(row,col)),shape=(2*nr,n),dtype=int)
        adj=A.T.dot(A)
        g=nx.convert_matrix.from_scipy_sparse_matrix(adj)

    if args.type==1:
        g=nx.gnm_random_graph(n,nr,seed=seed)
        adj=nx.adjacency_matrix(g)

    lcc=np.array(list(max(nx.connected_components(g), key=len)))
    n=len(lcc)
    eta=eta[:,lcc]
    nu=nu[:,lcc]
    G=G[lcc]

    X0=np.exp(-G)
    r=n
    if rank:
        r=np.linalg.matrix_rank(adj.toarray()[np.ix_(lcc,lcc)])
        if output:
            print("rank is ", r, "lcc is ", n)

    XD1,XD2,inds=get_drive(eta,nu,k,G,d0,nd)

    bif=-1
    Xs=np.array([])
    evals=np.array([])


    if quasi and r==n: #if r<n, steady state is not unique and numerical continuation is singular
        Xs,evals,bif=quasistatic(X0, eta, nu, k, XD1, XD2, 0, 100, steps, output)

    m=len(Xs)-1

    stop=timeit.default_timer()
    file=open(filebase+'out.dat','w')
    print(n,nr,nd,na,seed,steps,skip,d0,d1max, file=file)
    print('%.3f\t%i\t%i\t%i\t%i\t%i'%(stop-start, seed, n, r, bif, m), file=file)
    file.close()

    if output:
        print('%.3f\t%i\t%i\t%i\t%i\t%i'%(stop-start, seed, n, r, bif, m), flush=True)
        np.save(filebase+'Xs.npy',Xs[::skip])
        np.save(filebase+'evals.npy',evals[::skip])
