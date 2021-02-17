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
            k[2*i+1]=k[2*i]*np.exp(-deltaG)
        else:
            k[2*i+1]=np.random.exponential(scale=-deltaG)
            k[2*i]=k[2*i+1]*np.exp(deltaG)

    return eta,nu,k,G

def get_drive(eta,nu,k,G,d0,nd):
    n=len(G)
    inds=np.random.choice(np.arange(n),size=nd,replace=False)
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

def quasistatic (X0, eta, nu, k, XD1, XD2, ep0, ep1, dep0, depmin=1e-6, depmax=1e-2, epthrs=1e-3, output=True, stop=True):
    n=len(X0)
    eps=[]
    sols=[]
    evals=[]
    bif=0
    count=0
    SNnum=5
    dX=X0
    dep=dep0
    ep=ep0

    while ((ep1>ep0 and ep<=ep1) or (ep1<ep0 and ep>=ep1)):
        if output:
            print('%.6f\t\r'%((ep-ep0)/(ep1-ep0)),end='')

        if np.abs(dep)<depmin:
            if output:
                print('\nFailed to converge! ',ep)
            bif=-1
            break

        success,solx=steady(X0,eta,nu,k,(1+ep)*XD1,XD2)

        if success:
            mat=jac(0,solx,eta,nu,k,(1+ep)*XD1,XD2)
            eval,evec=np.linalg.eig(mat)

            #Check if solution changed more than desired
            if len(eps)>1 and (np.linalg.norm(solx-(sols[-1]+dX)) > 1e1*np.linalg.norm(dX) or np.min(np.abs(eval))/np.min(np.abs(evals[-1])) < 0.75):
                ep=eps[-1]
                if output:
                    print('\nChanged too much! decreasing step %.6f \t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'%(ep,dep,np.linalg.norm(solx-(sols[-1]+dX)),np.linalg.norm(dX),np.min(np.abs(eval)),np.min(np.abs(evals[-1]))), end='')
                mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
                dep=dep/2

                dX=-np.linalg.solve(mat,dep*XD1)
                X0=sols[-1]+dX
                ep=ep+dep
                continue

            #Check if Hopf
            if np.max(np.real(eval))>0 and np.abs(np.imag(eval[np.argmax(np.real(eval))]))>0:
                if output and bif==0:
                    print('\nHopf bifurcation!',ep)
                bif=1
                if stop:
                    break

            #Check if Saddle Node
            if  len(sols)>SNnum:
                ys=np.min(np.abs(evals[-SNnum:]),axis=1)
                xs=eps[-SNnum:]
                ym=ys[-1]
                xm=xs[-1]+(xs[-1]-xs[-3])/(ys[-1]-ys[-3])*ys[-3]
                x0=xs[0]
                y0=ys[0]

                #Check if a root fit to the smallest eigenvalue shows SN closer than threshold
                if(np.abs(xm-x0)<dep0):
                    xs=np.concatenate([xs,np.flip(xs)])
                    ys=np.concatenate([ys,np.flip(-ys)])
                    sol2=leastsq(lambda x: x[0]+x[1]*ys**2-xs,[xm,(xm-x0)/y0**2])
                    xn2=sol2[0][0]-ep

                    #Saddle-node detected! Look for the second branch
                    if xn2<epthrs:
                        sols.append(solx)
                        eps.append(ep)
                        evals.append(eval)
                        ind=np.argmin(np.abs(eval))
                        ev=np.real(evec[:,ind])
                        iev=np.real(np.linalg.inv(evec))[ind]
                        alpha=XD1.dot(iev)/hess(0,solx,eta,nu,k,(1+ep)*XD1,XD2).dot(ev).dot(ev).dot(iev)
                        if xn2*alpha>0:
                            X2=solx-4*ev*np.sqrt(xn2*alpha/2)
                            success2,sol2x=steady(X2,eta,nu,k,(1+ep)*XD1,XD2)
                            found=False

                            if success2 and (np.linalg.norm(solx-sol2x) > 0.01*np.linalg.norm(2*ev*np.sqrt(xn2*alpha/2))):
                                found=True
                                X0=sol2x

                            else:
                                X3=solx+4*ev*np.sqrt(xn2*alpha/2)
                                success3,sol3x=steady(X3,eta,nu,k,(1+ep)*XD1,XD2)
                                if success3 and (np.linalg.norm(solx-sol3x) > 0.01*np.linalg.norm(2*ev*np.sqrt(xn2*alpha/2))):
                                    found=True
                                    X0=sol3x
                                else:
                                    if output:
                                        print('\nSecond branch not found!\t%i\t%i\t%f\t%f\n'%(success2,success3,np.linalg.norm(solx-sol2x)/np.linalg.norm(2*ev*np.sqrt(xn2*alpha/2)),np.linalg.norm(solx-sol3x)/np.linalg.norm(2*ev*np.sqrt(xn2*alpha/2))),end='')
                                    if success2 and success3:
                                        mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
                                        dep=dep/2

                                        dX=-np.linalg.solve(mat,dep*XD1)
                                        X0=sols[-1]+dX
                                        ep=ep+dep

                            #Change branches and direction or break if branch is found
                            if found:
                                if bif==0:
                                    bif=2
                                if output:
                                    print('\nSaddle-node bifurcation!',ep)

                                success,solx=steady(X0,eta,nu,k,(1+ep)*XD1,XD2)
                                sols.append(solx)
                                mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
                                eval,evec=np.linalg.eig(mat)
                                eps.append(ep)
                                evals.append(eval)

                                if stop:
                                    break
                                else:
                                    dep=dep*-1
                                    dX=-np.linalg.solve(mat,dep*XD1)
                                    X0=sols[-1]+dX
                                    ep=ep+dep

                    #If the step is larger compared to predicted SN, decrease the step
                    if  np.abs(dep)>np.abs(xn2):
                        count=0
                        if output:
                            print('\nBifurcation expected, decreasing step! \t%.6f\t%.6f\t%.6f\n'%(ep,dep,xn2),end='')
                        ep=eps[-1]
                        mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
                        dep=dep/2
                        dX=-np.linalg.solve(mat,dep*XD1)
                        X0=sols[-1]+dX
                        ep=ep+dep
                        continue

            sols.append(solx)
            eps.append(ep)
            evals.append(eval)

            # Try to increase the step size if last 10 successful
            count=count+1
            if count/10==1:
                count=0
                if np.abs(dep)<depmax:
                    dep=dep*2

            # half the stepsize until the relative expected change is small
            dX=-np.linalg.solve(mat,dep*XD1)
            while np.max(np.abs(dX/solx)) > 1e-1:
                dep=dep/2
                dX=-np.linalg.solve(mat,dep*XD1)
                count=0

            X0=solx+dX
            ep=ep+dep

        #If solution is lost, decrease the step and try again
        else:
            count=0
            ep=eps[-1]
            if output:
                print('\nBranch lost! decreasing step %.6f %.6f\t\n'%(ep,dep), end='')
            mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
            dep=dep/2
            dX=-np.linalg.solve(mat,dep*XD1)
            X0=sols[-1]+dX
            ep=ep+dep

    return np.array(sols),np.array(eps),np.array(evals),bif

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
        Xs,epsilons,evals,bif=quasistatic(X0, eta, nu, k, XD1, XD2, 0, 100, 100/steps, output)
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
