#!/usr/bin/env python
import timeit
import argparse
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.optimize import root
from scipy.optimize import leastsq
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
from scipy.signal import find_peaks
from scipy.linalg import eig
from scipy.special import seterr

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

    return eta,nu,k,G,[]

def get_network_atoms(n,nr,na=0,natoms=0,verbose=False):

    eta=np.zeros((2*nr,n))
    nu=np.zeros((2*nr,n))
    k=np.zeros(2*nr)
    G=np.random.normal(loc=0, scale=1.0, size=n)
    atoms=np.random.randint(0,5,size=(n,natoms))

    i=0
    repeats=0
    while i<nr:
        reactants=np.random.choice(np.arange(n),size=np.random.randint(1,4),replace=False)
        rcounts=np.random.randint(1,3,size=len(reactants))

        success=False
        count=0
        if verbose:
            print('%d  %d %d\r'%(i, repeats, counts),end='')
        count=count+1
        products=[]
        tatoms=np.sum(rcounts[:,np.newaxis]*atoms[reactants],axis=0)
        for j in range(3):
            choices=np.setdiff1d(np.where([np.all(atoms[k]<=tatoms) for k in range(n)])[0], reactants)
            if len(choices) == 0:
                break
            prod=np.random.choice(choices,size=1,replace=False)
            products=products+[prod]
            tatoms=tatoms-atoms[prod]
            if np.sum(tatoms)==0:
                success=True
                break

        if success:
            products,pcounts=np.unique(products,return_counts=True)
            new_eta=np.zeros(n)
            new_nu=np.zeros(n)
            new_eta[reactants]=rcounts
            new_nu[products]=pcounts
            repeat=False
            for j in range(2*i):
                if np.all(new_eta==eta[j]) and np.all(new_nu==nu[j]):
                    repeat=True
            if not repeat:
                eta[2*i]=new_eta
                nu[2*i]=new_nu

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

                i=i+1
            else:
                repeats=repeats+1
    return eta,nu,k,G,atoms

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

def Sdot(rates):
    Jp=rates[::2]
    Jm=rates[1::2]
    return np.sum((Jp-Jm)*np.log(Jp/Jm))

def Wdot(X, G, XD1, XD2):
    return (XD1-XD2*X).dot(G+np.log(X))

def func(t, X, eta, nu, k, XD1, XD2):
    return XD1-XD2*X+rates(X,eta,nu,k).dot(eta-nu)

def jac(t,X,eta,nu,k,XD1,XD2):
    return -np.diag(XD2)+np.transpose((eta-nu)*rates(X,eta,nu,k)[:,np.newaxis]).dot(nu/X)

def hess(t,X,eta,nu,k,XD1,XD2):
    return np.tensordot(eta-nu,rates(X,eta,nu,k)[:,np.newaxis,np.newaxis]/(X[np.newaxis,:,np.newaxis]*X[np.newaxis,np.newaxis,:])*(nu[:,:,np.newaxis]*nu[:,np.newaxis,:]-nu[:,:,np.newaxis]*np.identity(len(X))[np.newaxis,:,:]), axes=(0,0))

def third(t,X,eta,nu,k,XD1,XD2):
    return np.tensordot(eta-nu,rates(X,eta,nu,k)[:,np.newaxis,np.newaxis,np.newaxis]/(X[np.newaxis,:,np.newaxis,np.newaxis]*X[np.newaxis,np.newaxis,:,np.newaxis]*X[np.newaxis,np.newaxis,np.newaxis,:])*(nu[:,:,np.newaxis,np.newaxis]*nu[:,np.newaxis,:,np.newaxis]*nu[:,np.newaxis,np.newaxis,:]-nu[:,:,np.newaxis,np.newaxis]*nu[:,np.newaxis,:,np.newaxis]*(np.identity(len(X))[np.newaxis,np.newaxis,:,:]+np.identity(len(X))[np.newaxis,:,np.newaxis,:]+np.identity(len(X))[np.newaxis,:,:,np.newaxis])+2*nu[:,:,np.newaxis,np.newaxis]*np.identity(len(X))[np.newaxis,:,:,np.newaxis]*np.identity(len(X))[np.newaxis,np.newaxis,:,:]), axes=(0,0))

def lcoeff(t,X,eta,nu,k,XD1,XD2,q,p,omega):
    A=jac(t,X,eta,nu,k,XD1,XD2)
    B=hess(t,X,eta,nu,k,XD1,XD2)
    C=third(t,X,eta,nu,k,XD1,XD2)
    v=np.linalg.solve(A,B@q@q.conjugate())
    w=np.linalg.solve(2*1j*omega*np.identity(len(X))-A,B@q@q)
    return 1/(2*omega)*np.real(np.vdot(p,C@q@q@q.conjugate()-2*B@q@v+B@q.conjugate()@w))

def steady(X0, eta, nu, k, XD1, XD2):
    sol=root(lambda x:func(0,x,eta,nu,k,XD1,XD2),x0=X0,jac=lambda x:jac(0,x,eta,nu,k,XD1,XD2), method='hybr', options={'xtol':1e-6,'diag':1/X0})
    if np.min(sol.x)>0 and sol.success:
        return True,sol.x
    else:
        return False,X0

def integrate(X0, eta, nu, k, XD1, XD2, t1, dt, maxcycles=100, output=False, maxsteps=1e6,cont=False):
    Xts=X0[:,np.newaxis]
    ts=np.array([0.])
    dts=np.array([])
    minds=[]
    m0=0
    state=-1
    stop=False
    dt0=dt/100
    dtmax=dt*1e6
    success=False
    try:
        while not stop or cont:

            try:
                if output:
                    print('%.6f\t%.6f\t%i\t%i\tlsoda\t\r'%(ts[-1]/t1, dt/t1, len(ts), len(minds)), end='',flush=True)
                sol=solve_ivp(func,(0,dt),Xts[:,-1],method='LSODA',dense_output=True,args=(eta, nu, k, XD1, XD2),rtol=1e-6,atol=1e-6*X0,jac=jac,max_step=dt,first_step=dt0/100)
                if sol.success:
                    dts=np.concatenate((dts,np.diff(sol.t)))
                    Xts=np.concatenate((Xts,sol.y[:,1:]),axis=1)
                    ts=np.concatenate((ts,ts[-1]+sol.t[1:]))
                else:
                    raise Exception(sol.message)
            except Exception as e:
                if output:
                    print('%.6f\t%.6f\t%i\t%i\tirk  \t%s\r'%(ts[-1]/t1, dt/t1, len(ts), len(minds),sol.message), end='',flush=True)
                sol=solve_ivp(func,(0,dt),Xts[:,-1],method='Radau',dense_output=True,args=(eta, nu, k, XD1, XD2),rtol=1e-6,atol=1e-6*X0,jac=jac,max_step=dt/10,first_step=dt0)
                if sol.success:
                    dts=np.concatenate((dts,np.diff(sol.t)))
                    Xts=np.concatenate((Xts,sol.y[:,1:]),axis=1)
                    ts=np.concatenate((ts,ts[-1]+sol.t[1:]))
                else:
                    raise Exception(sol.message)

            #update timesteps
            tscales=np.max(np.abs(np.diff(Xts,axis=1)/dts/Xts[:,1:]),axis=0)
            tinds=np.where(ts>ts[-1]/2)[0]
            dt=np.min([np.mean(10/tscales[tinds[:-1]]),100*dt0,ts[-1]/2,dtmax])
            dt=np.min([t1-ts[-1],dt])
            dt0=np.min([dt,dts[-2]])

            #check for stopping
            if len(ts)>maxsteps:
                if output:
                    print('\nFailed to find state in maxsteps!',len(ts))
                break
            if ts[-1]>=t1:
                if output:
                    print('\nFailed to find state before maxtime',t1)
                break

            if not stop:
                #check for steady state
                success,solx=steady(Xts[:,-1],eta,nu,k,XD1,XD2)
                if success and np.linalg.norm((solx-Xts[:,-1])/solx)<1e-2:
                    ev,evec=np.linalg.eig(jac(0,solx,eta,nu,k,XD1, XD2))
                    if np.max(np.real(ev))<0:
                        if output:
                            print('\nFound steady state!')
                        Xts[:,-1]=solx
                        m0=len(ts)-1
                        state=0
                        stop=True
                        success=True

                #check for oscillating state
                norms=np.linalg.norm(Xts,axis=0)
                minds=find_peaks(norms)[0]
                if len(minds)>maxcycles:
                    max=np.max(norms[minds[-maxcycles:]])
                    min=np.min(norms[minds[-maxcycles:]])
                    minds=find_peaks(norms,prominence=(max-min)/2)[0]
                    if len(minds)>=maxcycles:
                        dt=10*np.mean(np.diff(ts[minds[-maxcycles:]]))
                        max=np.max(norms[minds[-maxcycles]:])
                        min=np.min(norms[minds[-maxcycles]:])

                        sol2=leastsq(lambda x: norms[minds[-maxcycles:]]-x[0]+x[1]*ts[minds[-maxcycles:]],[np.mean(norms[minds[-maxcycles:]]),0])
                        if np.abs(sol2[0][1]*(ts[-1]-ts[minds[-maxcycles]])/(max-min)) < 0.1:
                            if output:
                                print('\nFound oscillating state!')
                            m0=minds[-maxcycles]
                            state=1
                            stop=True
                            success=True

    except Exception as e:
        raise e

    return ts,Xts,success,m0,state

def quasistatic (X0, eta, nu, k, XD1, XD2, ep0, ep1,ep, dep0, depmin=1e-12, depmax=1e-2, epthrs=1e-4, stepsmax=1e5, output=True, stop=True):
    n=len(X0)
    eps=[]
    sols=[]
    evals=[]
    bif=0
    count=0
    SNnum=5
    dX=X0
    dep=dep0
    steps=0
    scount=0
    depmax=np.min([np.abs(dep0),depmax])

    while ((ep<=ep1 and ep>=ep0)) and steps<stepsmax:
        steps=steps+1
        if output>0:
            print('%.6f\t%.6e\t%i\t\r'%((ep-ep0)/(ep1-ep0),dep,steps),end='')

        if np.abs(dep)<depmin:
            if output>1:
                print('\nFailed to converge!\t\t%f\t%e\n'%(ep,dep),end='')
            bif=-1
            break

        success,solx=steady(X0,eta,nu,k,(1+ep)*XD1,XD2)

        #If solution is lost, decrease the step and try again
        if not success:
            count=0
            ep=eps[-1]
            if output>2:
                print('\nBranch lost! \t%.6f\t%.6f\t%i\n'%(ep,dep,len(sols)), end='')
            mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
            dep=dep/4
            dX=-np.linalg.solve(mat,dep*XD1)
            X0=sols[-1]+dX
            ep=ep+dep

        else:
            mat=jac(0,solx,eta,nu,k,(1+ep)*XD1,XD2)
            eval,evec=eig(mat)

            #Check if Hopf (complex eigenvalue with smallest real part changes sign)
            omega=np.imag(eval[np.argmin(np.abs(np.real(eval)))])
            if len(evals)>1 and omega!=0.:

                if (np.count_nonzero(np.real(evals[-1])>0)==0 and np.count_nonzero(np.real(eval)>0)>0) or (np.count_nonzero(np.real(evals[-1])>0)>0 and np.count_nonzero(np.real(eval)>0)==0):
                    bif=1
                    #locate the bifurcation point and find the lyapunov coefficient
                    hsol=root(lambda x:np.max(np.real(eig(jac(0,steady(X0,eta,nu,k,(1+x)*XD1,XD2)[1],eta,nu,k,(1+x)*XD1,XD2))[0])),x0=ep)
                    eph=hsol.x[0]
                    X0h=steady(X0,eta,nu,k,(1+eph)*XD1,XD2)[1]
                    ev,lvec,rvec=eig(jac(0,X0h,eta,nu,k,(1+eph)*XD1, XD2),left=True,right=True)

                    sols.append(X0h)
                    eps.append(eph)
                    evals.append(ev)

                    ind=np.argmin(np.abs(np.real(eval)))
                    omega=np.imag(eval[ind])

                    q=rvec[:,ind]
                    p=lvec[:,ind]/np.vdot(rvec[:,ind],lvec[:,ind])
                    if omega<0:
                        omega=-omega
                        q=q.conjugate()
                        p=p.conjugate()
                    l=lcoeff(0,X0h,eta,nu,k,(1+eph)*XD1,XD2,q,p,omega)
                    if l<0:
                        bif=1
                        if output>1:
                            print('\nSupercritical Hopf bifurcation!\t\t%f\n'%(ep),end='')
                    else:
                        bif=3
                        if output>1:
                            print('\nSubcritical Hopf bifurcation!\t\t%f\n'%(ep),end='')

                    if stop:
                        break

                sols.append(solx)
                eps.append(ep)
                evals.append(eval)
                dX=-np.linalg.solve(mat,dep*XD1)
                X0=sols[-1]+dX
                ep=ep+dep
                continue

            #Check if number of real eigenvalues changed
            if len(evals)>0 and (np.count_nonzero(np.real(evals[-1])>0)!=np.count_nonzero(np.real(eval)>0)>0):
                if output>2:
                    print('\nMissed bifurcation!\t%f'%(ep))
                ep=eps[-1]
                mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
                dep=dep/4
                dX=-np.linalg.solve(mat,dep*XD1)
                X0=sols[-1]+dX
                continue

            #Check if Saddle Node
            if  scount>2*SNnum and omega==0:
                ys=np.min(np.abs(evals[-SNnum:]),axis=1)
                xs=eps[-SNnum:]
                xm=xs[-1]-(xs[-1]-xs[-3])/(ys[-1]-ys[-3])*ys[-1]
                x0=xs[0]
                y0=ys[0]
                xs=np.concatenate([xs,np.flip(xs)])
                ys=np.concatenate([ys,np.flip(-ys)])
                sol2=leastsq(lambda x: x[0]+x[1]*ys**2-xs,[xm,(xm-x0)/y0**2])
                sep=sol2[0][0]
                #If SN appears close, look for a root
                if (np.abs(sep-xs[-1])<np.abs(epthrs)) and (sep-xs[-1])/dep>0:
                    mat=jac(0,solx,eta,nu,k,(1+sep)*XD1,XD2)
                    eval,lvec,rvec=eig(mat,left=True)

                    ind=np.argmin(np.abs(eval))
                    ev=np.real(rvec[:,ind])
                    iev=np.real(lvec[:,ind])
                    ev=ev/ev.dot(ev)
                    iev=iev/iev.dot(ev)

                    alpha=np.real(XD1.dot(iev)/(hess(0,solx,eta,nu,k,(1+sep)*XD1,XD2).dot(ev).dot(ev).dot(iev)))

                    if (sep-ep)*alpha>0:
                        success2,sol2x=steady(solx-10*ev*np.sqrt((sep-ep)*alpha/2),eta,nu,k,(1+ep)*XD1,XD2)
                        success3,sol3x=steady(solx+10*ev*np.sqrt((sep-ep)*alpha/2),eta,nu,k,(1+ep)*XD1,XD2)
                        eval2,evec2=np.linalg.eig(jac(0,sol2x,eta,nu,k,(1+ep)*XD1,XD2))
                        eval3,evec3=np.linalg.eig(jac(0,sol3x,eta,nu,k,(1+ep)*XD1,XD2))
                        found=False
                        c1=np.count_nonzero(np.real(eval)>0)
                        c2=np.count_nonzero(np.real(eval2)>0)
                        c3=np.count_nonzero(np.real(eval3)>0)

                        if success2 and c1!=c2:
                            found=True
                            X0=sol2x

                        elif success3 and c1!=c3:
                            found=True
                            X0=sol3x

                        if found:
                            scount=0
                            if bif==0:
                                bif=2
                            if output>1:
                                # print('\nSaddle-node bifurcation!\t%.6f\n'%(ep),end='')
                                print('\nSaddle-node bifurcation!\t%.6f\t%.6e\t%i\t%i\t%f\t%f\t%i\t%i\t%i\t\n'%(ep,dep,success2,success3,np.abs((solx-sol2x).dot(iev))/np.sqrt((sep-ep)*alpha/2),np.abs((solx-sol3x).dot(iev))/np.sqrt((sep-ep)*alpha/2),c1,c2,c3),end='')

                            mat=jac(0,solx,eta,nu,k,(1+ep)*XD1,XD2)
                            eval,evec=np.linalg.eig(mat)
                            sols.append(solx)
                            eps.append(ep)
                            evals.append(eval)
                            mat=jac(0,X0,eta,nu,k,(1+ep)*XD1,XD2)
                            eval,evec=np.linalg.eig(mat)
                            sols.append(X0)
                            eps.append(ep)
                            evals.append(eval)

                            if stop:
                                break
                            else:
                                dep=-dep
                                dX=-np.linalg.solve(mat,dep*XD1)
                                X0=sols[-1]+dX
                                ep=ep+dep
                                continue
                        else:
                            if output>2:
                                print('\nSecond branch not found!\t%.6f\t%.6e\t%i\t%i\t%f\t%f\t%i\t%i\t%i\t\n'%(ep,dep,success2,success3,np.abs((solx-sol2x).dot(iev))/np.sqrt((sep-ep)*alpha/2),np.abs((solx-sol3x).dot(iev))/np.sqrt((sep-ep)*alpha/2),c1,c2,c3),end='')

                            ep=eps[-1]
                            mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
                            dep=dep/2
                            dX=-np.linalg.solve(mat,dep*XD1)
                            X0=sols[-1]+dX
                            ep=ep+dep
                            continue

                    #If the step is larger compared to predicted SN, decrease the step
                    if  np.abs(dep)>np.abs((sep-ep)) and (sep-ep)/dep>0:
                        count=0
                        if output>2:
                            print('\nBifurcation expected! \t%.6f\t%.6f\t%.6f\n'%(ep,dep,(sep-ep)),end='')
                        ep=eps[-1]
                        mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
                        dep=dep/2
                        dX=-np.linalg.solve(mat,dep*XD1)
                        X0=sols[-1]+dX
                        ep=ep+dep
                        continue

            #Check if solution changed more than desired
            if len(eps)>1 and (np.linalg.norm(solx-(sols[-1]+dX))/np.linalg.norm(dX) > 1e1 or (np.min(np.abs(eval))/np.min(np.abs(evals[-1])) < 0.9 and scount>SNnum)):
                ep=eps[-1]
                if output>2:
                    print('\nChanged too much!\t%.6f \t%.6e\t%.3f\t%.3f\t%i\n'%(ep,dep,np.linalg.norm(solx-(sols[-1]+dX))/np.linalg.norm(dX),np.min(np.abs(eval))/np.min(np.abs(evals[-1])),np.count_nonzero(np.where(np.real(eval)<0))!=np.count_nonzero(np.where(np.real(evals[-1])<0))), end='')
                mat=jac(0,sols[-1],eta,nu,k,(1+ep)*XD1,XD2)
                dep=dep/4

                dX=-np.linalg.solve(mat,dep*XD1)
                X0=sols[-1]+dX
                ep=ep+dep
                continue


            sols.append(solx)
            eps.append(ep)
            evals.append(eval)
            scount=scount+1

            # Try to increase the step size if last 10 successful
            count=count+1
            if count/10==1:
                count=0
                if np.abs(2*dep)<=depmax:
                    dep=dep*2
                else:
                    dep=np.sign(dep)*depmax

            # half the stepsize until the relative expected change is small
            dX=-np.linalg.solve(mat,dep*XD1)
            while np.max(np.abs(dX/solx)) > 1e-1:
                dep=dep/2
                dX=-np.linalg.solve(mat,dep*XD1)
                count=0

            X0=solx+dX
            ep=ep+dep



    if steps>=stepsmax:
        bif=-1

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
    parser.add_argument("--dep", type=int, default=1e-2, dest='dep', help='Step size for driving')
    parser.add_argument("--na", type=int, default=0, dest='na', help='Number of autocatalytic reactions')
    parser.add_argument("--skip", type=int, default=10, dest='skip', help='Steps to skip for output')
    parser.add_argument("--output", type=int, default=0, dest='output', help='1 for matrix output, 0 for none')
    parser.add_argument("--quasistatic", type=int, default=1, dest='quasi', help='1 for quasistatic')
    parser.add_argument("--integrate", type=int, default=1, dest='integ', help='1 for integrate')
    parser.add_argument("--rank", type=int, default=1, dest='rank', help='1 for rank calculation')
    args = parser.parse_args()
    n=args.n
    nr=args.nr
    nd=args.nd
    na=args.na
    filebase=args.filebase
    output=args.output
    quasi=args.quasi
    integ=args.integ
    rank=args.rank
    seed=args.seed
    dep=args.dep
    skip=args.skip
    d0=args.d0
    d1min=1
    d1max=args.dmax
    np.random.seed(seed)
    np.seterr(all='ignore')
    seterr(all='ignore')

    #We should find the lcc of the network and discard the rest.
    start=timeit.default_timer()
    eta,nu,k,G,atoms=get_network_atoms(n,nr,na)

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
    state=-1
    Xs=np.array([])
    evals=np.array([])
    epsilon=0
    sd1=0
    sd2=0
    wd1=0
    wd2=0
    dn1=0
    dn2=0

    if quasi and r==n: #if r<n, steady state is not unique and continuation is singular
        Xs,epsilons,evals,bif=quasistatic(X0, eta, nu, k, XD1, XD2, 0, d1max, 0, dep, output=output,stop=True)
        sd1=Sdot(rates(Xs[-1],eta,nu,k))
        wd1=Wdot(Xs[-1], G, (1+epsilons[-1])*XD1, XD2)

        #following a bifurcation, integrate the system
        if bif>0 and integ:
            try:
                X0=Xs[-1]
                epsilon=epsilons[-1]+1e-2
                ev,evec=eig(jac(0,X0,eta,nu,k,(1+epsilon)*XD1, XD2))
                tscale=2*np.pi/np.abs(np.real(ev[np.argmin(np.abs(np.real(ev)))]))
                dt=100/np.max(np.abs(func(0,X0,eta,nu,k,(1+epsilon)*XD1, XD2)/X0))

                if output:
                    print('\nIntegrating',epsilon,tscale,dt)
                ts,Xts,success,m0,state=integrate(X0,eta,nu,k,(1+epsilon)*XD1,XD2,1e4*tscale,dt,output=output)
                print(m0,len(ts))
                sd2=np.sum(np.diff(ts)[m0-1:]*[Sdot(rates(Xts[:,i],eta,nu,k)) for i in range(m0,len(ts))])/ np.sum(np.diff(ts)[m0-1:])
                wd2=np.sum(np.diff(ts)[m0-1:]*[Wdot(Xts[:,i], G, (1+epsilon)*XD1, XD2) for i in range(m0,len(ts))])/ np.sum(np.diff(ts)[m0-1:])
                dn1=np.sum(np.diff(ts)[m0-1:]*[np.linalg.norm(Xts[:,i]-X0) for i in range(m0,len(ts))])/ np.sum(np.diff(ts)[m0-1:])
                dn2=np.sum(np.diff(ts)[m0-1:]*[np.linalg.norm((Xts[:,i]-X0)/X0) for i in range(m0,len(ts))])/ np.sum(np.diff(ts)[m0-1:])
            except Exception as err:
                print('Error integrating seed %i\t%i\t%i\t%i\t%i\n'%(seed,n,nr,nd,na),end='')
                print(str(err))

    stop=timeit.default_timer()
    file=open(filebase+'out.dat','w')
    epsilon=epsilons[-1]
    print(n,nr,nd,na,seed,d0,d1max, file=file)
    print('%.3f\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%i'%(stop-start, seed, n, r, bif, epsilon, sd1, sd2, wd1, wd2, dn1, dn2, state), file=file)
    file.close()

    if output:
        print('\n%.3f\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%i'%(stop-start, seed, n, r, bif, epsilon, sd1, sd2, wd1, wd2, dn1, dn2, state))
        np.save(filebase+'Xs.npy',Xs[::skip])
        np.save(filebase+'epsilons.npy',epsilons[::skip])
        np.save(filebase+'evals.npy',evals[::skip])
