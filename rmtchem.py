#!/usr/bin/env python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.integrate import ode
import timeit
import argparse
import sys
from progressbar import *
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.optimize import leastsq
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
    inds=np.random.choice(np.arange(n),size=nd,replace=False)
    # inds=np.argsort(G)[-nd:] #drive the most stable species, to avoid large concentration ratios
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
    # sol=root(lambda x:func(0,np.exp(x),eta,nu,k,XD1,XD2),x0=np.log(X0),jac=lambda x:jac(0,np.exp(x),eta,nu,k,XD1,XD2)*np.exp(x), method='hybr', options={'xtol':1e-6})
    # return sol.success,np.exp(sol.x)
    sol=root(lambda x:func(0,x,eta,nu,k,XD1,XD2),x0=X0,jac=lambda x:jac(0,x,eta,nu,k,XD1,XD2), method='hybr', options={'xtol':1e-8})
    return sol.success,sol.x

def integrate(X0, eta, nu, k, XD1, XD2, t1, dt, prog=False):
    n=len(X0)
    success=1
    if prog:
        pbar=ProgressBar(widgets=['Integration: ', Percentage(),Bar(), ' ', ETA()], maxval=t1)
        pbar.start()
    Xs=np.zeros((int(t1/dt),n))
    rode=ode(func,jac).set_integrator('lsoda',rtol=1e-6,atol=1e-8,max_step=dt)
    rode.set_initial_value(X0, 0)
    rode.set_f_params(eta, nu, k, XD1, XD2)
    rode.set_jac_params(eta, nu, k, XD1, XD2)
    Xs[0]=X0
    for n in range(1,int(t1/dt)):
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

def snfunc(X,Xs,epsilons,alpha,beta,evecs,ind):
    X0=X[:-1]
    epsilon0=X[-1]
    inds=np.setdiff1d(np.arange(len(X0)),[ind])
    S0=(Xs-X0)
    S1=(((np.abs((epsilons-epsilon0)*alpha)/2)**0.5)[:,np.newaxis]*evecs[:,ind])
    S2=-(epsilons-epsilon0)[:,np.newaxis]*np.sum(beta[inds]*evecs[:,inds],axis=1)
    # S1=0
    S2=0
    return np.sum(np.linalg.norm(S0+S1+S2,axis=1))

#We should have epsilon passed rather than XD1s and XD2s
def quasistatic (X0, eta, nu, k, XD1, XD2, epsilon0, epsilon1, steps, output=True, stop=True):
    n=len(X0)
    nr=int(len(nu)/2)

    epsilons=[]
    sols=[]
    evals=[]
    depsilon=(epsilon1-epsilon0)/steps
    epsilon=0

    nu2=getNu2(nu)
    drives=np.zeros(n)
    drives[np.where(XD1!=0)[0]]=1
    bif=0
    count=0
    SNnum=5
    dX=X0
    depmin=(epsilon1-epsilon0)/steps/1e3
    epthrs=(epsilon1-epsilon0)/steps

    while epsilon<epsilon1:
        if output:
            print('%.4f\t\r'%((epsilon-epsilon0)/(epsilon1-epsilon0)),end='')

        success,solx=steady(X0,eta,nu,k,(1+epsilon)*XD1,XD2)
        if success:
            mat=jac(0,solx,eta,nu,k,(1+epsilon)*XD1,XD2)
            eval,evec=np.linalg.eig(mat)

            #Check if solution changed more than expected
            if len(epsilons)>1 and np.linalg.norm(solx-(sols[-1]+dX)) > 1.1*np.linalg.norm(dX):
                epsilon=epsilons[-1]
                if output:
                    print('\nChanged branches! decreasing step %.4f \t%.4f\t%.4f\t%.4f\n'%(epsilon,depsilon,np.linalg.norm(solx-(sols[-1]+dX)),np.linalg.norm(dX)), end='')
                mat=jac(0,sols[-1],eta,nu,k,(1+epsilon)*XD1,XD2)
                depsilon=depsilon/1.5

                if depsilon<(epsilon1-epsilon0)/steps/1000:
                    if output:
                        print('\nFailed to converge! ',epsilon)
                    bif=-1
                    break
                dX=-np.linalg.solve(mat,depsilon*XD1)
                X0=sols[-1]+dX
                epsilon=epsilon+depsilon
                continue

            #Check if Hopf
            if np.max(np.real(eval))>0 and np.abs(np.imag(eval[np.argmax(np.real(eval))]))>0:
                if output and bif==0:
                    print('\nHopf bifurcation!',epsilon)
                bif=1
                if stop:
                    break

            sols.append(solx)
            epsilons.append(epsilon)
            evals.append(eval)


            #Check if Saddle Node
            if  len(sols)>SNnum:
                #Fit both smallest eigenvalue and norms to quadratic
                ys=np.min(np.abs(evals[-SNnum:]),axis=1)
                xs=epsilons[-SNnum:]
                sol=leastsq(lambda x: x[0]+x[1]*ys**2-xs,[xs[-1],(xs[-1]-xs[0])/ys[0]**2])
                ys=np.linalg.norm(sols[-SNnum:]/sols[-SNnum],axis=1)
                xs=epsilons[-SNnum:]
                sol2=leastsq(lambda x: x[0]+x[1]*ys+x[2]*ys**2-xs,[xs[-1],(xs[-1]-xs[0])/ys[0],0])
                fn=np.linalg.norm(sol[0][0]+sol[0][1]*ys**2-xs)
                xn1=sol[0][0]-epsilon
                xn2=sol2[0][0]-sol2[0][1]**2/(4*sol2[0][2])-epsilon
                if depsilon>0.5*np.min(np.abs([xn1,xn2])):
                    depsilon=depsilon/1.5

                if np.min(np.abs(eval))<1e-2 and xn1<epthrs and xn2<epthrs and xn1>-depsilon and xn2>-depsilon:
                    if bif==0:
                        bif=2
                    if output:
                        print('\nSaddle-node bifurcation!',epsilon)
                    break

            # Try to increase the step size if last 10 successful
            count=count+1
            if count/10==1:
                count=0
                if depsilon<(epsilon1-epsilon0)/steps/2:
                    depsilon=depsilon*1.5

            # half the stepsize until the relative change is small
            dX=-np.linalg.solve(mat,depsilon*XD1)
            while np.max(np.abs(dX/solx)) > 1e-1:
                depsilon=depsilon/1.5
                dX=-np.linalg.solve(mat,depsilon*XD1)
                count=0

            if depsilon<depmin:
                if output:
                    print('\nFailed to converge! ',epsilon)
                bif=-1
                break

            X0=solx+dX
            epsilon=epsilon+depsilon

        else:
            epsilon=epsilons[-1]
            if output:
                print('\nBranch lost! decreasing step %.4f %.4f\t\n'%(epsilon,depsilon), end='')
            mat=jac(0,sols[-1],eta,nu,k,(1+epsilon)*XD1,XD2)
            depsilon=depsilon/1.5


            if depsilon<depmin:
                if output:
                    print('\nFailed to converge! ',epsilon)
                bif=-1
                break

            dX=-np.linalg.solve(mat,depsilon*XD1)
            X0=sols[-1]+dX
            epsilon=epsilon+depsilon

    return np.array(sols),np.array(epsilons),np.array(evals),bif

if __name__ == "__main__":
    #Command line arguments
    parser = argparse.ArgumentParser(description='Random chemical reaction networks.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output')
    parser.add_argument("--n", type=int, default=10, dest='n', help='Number of species')
    parser.add_argument("--nr", type=int, default=20, dest='nr', help='Number of reactions')
    parser.add_argument("--nd", type=int, default=1, dest='nd', help='Number of drives')
    parser.add_argument("--dmax", type=float, default=100, dest='dmax', help='Maximum drive')
    parser.add_argument("--type", type=int, default=0, dest='type', help='Type of adjacency matrix. 0 for chemical networks, 1 for ER networks.')
    parser.add_argument("--d0", type=float, default=1e6, dest='d0', help='Drive timescale')
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

    bif=-3
    Xs=np.array([])
    evals=np.array([])
    epsilon=0


    if quasi and r==n: #if r<n, steady state is not unique and numerical continuation is singular
        Xs,epsilons,evals,bif=quasistatic(X0, eta, nu, k, XD1, XD2, 0, 100, steps, output)
        epsilon=epsilons[-1]

    stop=timeit.default_timer()
    file=open(filebase+'out.dat','w')
    print(n,nr,nd,na,seed,steps,skip,d0,d1max, file=file)
    print('%.3f\t%i\t%i\t%i\t%i\t%f'%(stop-start, seed, n, r, bif, epsilon), file=file)
    file.close()

    if output:
        print('%.3f\t%i\t%i\t%i\t%i\t%f'%(stop-start, seed, n, r, bif, epsilon), flush=True)
        np.save(filebase+'Xs.npy',Xs[::skip])
        np.save(filebase+'epsilons.npy',epsilons[::skip])
        np.save(filebase+'evals.npy',evals[::skip])
