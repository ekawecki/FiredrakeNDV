"""This program is currently set up to approximate the uniformly convex solution of various Monge--Ampere type problems  on the unit disk, with Dirichlet boundary conditions.
    
    If you do use this code, please acknowledge the authors Ellya Kawecki, Omar  & Tristan Pryer.
    
    This code is compatible with an earlier release of FEniCS than the latest available version, so they may be some issues with domains, etc, please contact me if you have any problems. To download FEniCS, simply google "fenics", I recommend using either a Linux or Mac system.
    
    All options are at the top of the file are integer switches, change these to
    change the options.
    """

__author__ = "Ellya Kawecki (ellya.kawecki@queens.ox.ac.uk / kawecki@maths.ox.ac.uk)"
__date__ = "22-01-2015"
__copyright__ = "Copyright (C) 2015 Ellya Kawecki"
__license__  = "GNU LGPL Version 2.1 (or whatever is the latest one)"

#import Gnuplot, Gnuplot.funcutils
from firedrake import *
import numpy as np
from time import time

# Specifying choice of benchmark solution and problem, number of mesh refinements, maximum Newton method iterations, Newton method tolerance, choice of domain, and if we are considering the problem of prescribed Gaussian curvature with an artificial exponential term, Gaussian curvature, or the Weingarten equation
prob = 2
refinementsno = 7
newtonitermax = 40
newtontol = 1e-12
disk = True
quad_deg = 75
lessadvec = False
Gauss = False
weingarten = False
nitsche = False
r = 10
dataoutput = True
def det(M):
    deter = M[0][0]*M[1][1] - M[0][1]*M[1][0]
    return deter

def Newton(idx,deg,r):
    if disk == True:
        mesh = Mesh("Meshes/quasiunifrefdisk_%i.msh" % idx)
    else:
        mesh = UnitSquareMesh(2**(idx+1),2**(idx+1))
        bdry_indicator = 0.0
    # Implementing quadratic domain approximation if the domain has a curved boundary portion.
    if disk == True:
        V = FunctionSpace(mesh, "CG", 2)
        bdry_indicator = Function(V)
        bc = DirichletBC(V, Constant(1.0), 1)
        bc.apply(bdry_indicator)
        VV = VectorFunctionSpace(mesh, "CG", 2)
        T = Function(VV)
        T.interpolate(SpatialCoordinate(mesh))
        # Quadratic interpolation of the unit disk chart x/|x|
        T.interpolate(conditional(abs(1-bdry_indicator) < 1e-10, T/sqrt(inner(T,T)), T))
        mesh = Mesh(T)

    # unit normal
    n = FacetNormal(mesh)
    
    # finite element space
    S = VectorFunctionSpace(mesh,"CG",deg,dim = 4)
    
    # defining maximum mesh size
    DG = FunctionSpace(mesh, "DG", 0);
    h = sqrt(max(interpolate(CellVolume(mesh), DG).vector()[:]))
    hmin = sqrt(min(interpolate(CellVolume(mesh), DG).vector()[:]))
    # defining local mesh size
    hfct = sqrt(CellVolume(mesh))
    
    # coordinate functions
    x, y = SpatialCoordinate(mesh)
    if prob == 0:
        alp = 0.05
        convup = 10.0
        rho = pow(x,2)+pow(y,2)
        u = convup*(0.5*(rho-1)-alp*0.25*sin(pi*rho))
        du0 = convup*(x - alp*0.5*pi*x*cos(pi*rho))
        du1 = convup*(y - alp*0.5*pi*y*cos(pi*rho))
        d2udxx = convup*(Constant(1.0)-alp*0.5*pi*cos(pi*rho)+alp*pi**2*x**2*sin(pi*rho))
        d2udxy = convup*(Constant(0.0)+alp*pi**2*x*y*sin(pi*rho))
        d2udyy = convup*(Constant(1.0)-alp*0.5*pi*cos(pi*rho)+alp*pi**2*y**2*sin(pi*rho))
        g = u
    elif prob == 1:
        mult = 1.0
        alp = -0.05
        rho = pow(x,2)+pow(y,2)
        u = mult*(2.5*(rho-1)+alp*(cos(pi*rho)+1.0))
        du0 = mult*(5.0*x-2.0*alp*x*pi*sin(pi*rho))
        du1 = mult*(5.0*y-2.0*alp*y*pi*sin(pi*rho))
        d2udxx = mult*(5.0-2.0*alp*pi*sin(pi*rho)-4.0*alp*x*pi*x*pi*cos(pi*rho))
        d2udxy = mult*(-4.0*alp*pi*x*y*pi*cos(pi*rho))
        d2udyy = mult*(5.0-2.0*alp*pi*sin(pi*rho)-4.0*alp*y*pi*y*pi*cos(pi*rho))
        g = u
    elif prob == 2:
        rho = pow(x,2) + pow(y,2)
        u = -sqrt(2-rho)+1.0
        du0 = x/sqrt(2-rho)
        du1 = y/sqrt(2-rho)
        d2udxx = -(-2+pow(y,2))/(pow((2-rho),1.5))
        d2udyy = -(-2+pow(x,2))/(pow((2-rho),1.5))
        d2udxy = -(-x*y)/(pow(2-rho,1.5))
        g = u

    gradu = as_vector([du0, du1])
    Hessu = as_matrix([[d2udxx, d2udxy], [d2udxy, d2udyy]])
    if Gauss == True:
        f = det(Hessu)/(pow(1+pow(du0,2)+pow(du1,2),2))
    elif weingarten == True:
        f = det(Hessu)/(dot(gradu,gradu)+1)**2+((1+du0**2+du1**2)*(d2udxx+d2udyy)-du0*du0*d2udxx-2.0*du0*du1*d2udxy-du1*du1*d2udyy)/pow((1+du0*du0+du1*du1),3.0/2.0)
    elif lessadvec == True:
        f = exp(-u)*det(Hessu)/(pow(1+pow(du0,2)+pow(du1,2),2))
    else:
        f = det(Hessu)
    bc0 = DirichletBC(S.sub(0), Constant(0.0), 1)
    def Model(Dxx,Dxy,Dyy,Dx,Dy,uinit):
        #Splitting the solution and test function into it's separate components for defining our nonlinear problem.
        uh, H00, H11, H01 = TrialFunction(S)
        vh, phi00, phi11, phi01 = TestFunction(S)
        

        #Defining our Hessian to make our nonlinear form more compact
        Hessuh = as_matrix([[H00, H01], [H01, H11]])
        graduh = as_vector([uh.dx(0), uh.dx(1)])
        Hessn = as_matrix([[Dxx, Dxy], [Dxy, Dyy]])
        gradn = as_vector([Dx, Dy])
        un = uinit

        #Defining our nonlinear problem F

        # prescribed gauss curvature equation
        if Gauss == True:
            A = ( H00 * phi00 + uh.dx(0) * phi00.dx(0) )*dx(mesh, degree = quad_deg)\
                + ( H11 * phi11 + uh.dx(1) * phi11.dx(1) )*dx(mesh, degree = quad_deg)\
                + ( H01 * phi01 + uh.dx(0) * phi01.dx(1) )*dx(mesh, degree = quad_deg)\
                - (phi00 * n[0]) * (uh.dx(0)) * ds(mesh, degree = quad_deg)\
                - (phi11 * n[1]) * (uh.dx(1)) * ds(mesh, degree = quad_deg)\
                - (phi01 * n[1]) * (uh.dx(0)) * ds(mesh, degree = quad_deg)\
                + (4.0*det(Hessn)/pow(1+pow(Dx,2)+pow(Dy,2),3))*(Dx*uh.dx(0)+Dy*uh.dx(1)) * vh * dx(mesh, degree = quad_deg)\
                - ((Dyy*H00+Dxx*H11-2*Dxy*H01)/pow(1+pow(Dx,2)+pow(Dy,2),2)) * vh * dx(mesh, degree = quad_deg)
            L = - (dmp*f+det(Hessn)/pow(1+pow(Dx,2)+pow(Dy,2),2)+(1-dmp)*det(Hessn)/pow(1+pow(Dx,2)+pow(Dy,2),2))* vh * dx(mesh, degree = quad_deg)\
                + (4.0*det(Hessn)*(pow(Dx,2)+pow(Dy,2))/pow(1+pow(Dx,2)+pow(Dy,2),3)) * vh * dx(mesh, degree = quad_deg)
        elif lessadvec == True:
            A = ( H00 * phi00 + uh.dx(0) * phi00.dx(0) )*dx(mesh, degree = quad_deg)\
                + ( H11 * phi11 + uh.dx(1) * phi11.dx(1) )*dx(mesh, degree = quad_deg)\
                + ( H01 * phi01 + uh.dx(0) * phi01.dx(1) )*dx(mesh, degree = quad_deg)\
                - (phi00 * n[0]) * (uh.dx(0)) * ds(mesh, degree = quad_deg)\
                - (phi11 * n[1]) * (uh.dx(1)) * ds(mesh, degree = quad_deg)\
                - (phi01 * n[1]) * (uh.dx(0)) * ds(mesh, degree = quad_deg)\
                + (4.0*det(Hessn)/pow(1+pow(Dx,2)+pow(Dy,2),3))*(Dx*uh.dx(0)+Dy*uh.dx(1)) * vh * dx(mesh, degree = quad_deg)\
                - ((Dyy*H00+Dxx*H11-2*Dxy*H01)/pow(1+pow(Dx,2)+pow(Dy,2),2)) * vh * dx(mesh, degree = quad_deg)\
                + f*exp(un)*uh * vh * dx(mesh, degree = quad_deg)
            L = - (dmp*f*exp(un)+det(Hessn)/pow(1+pow(Dx,2)+pow(Dy,2),2)+(1-dmp)*det(Hessn)/pow(1+pow(Dx,2)+pow(Dy,2),2))* vh * dx(mesh, degree = quad_deg)\
                + (4.0*det(Hessn)*(pow(Dx,2)+pow(Dy,2))/pow(1+pow(Dx,2)+pow(Dy,2),3)) * vh * dx(mesh, degree = quad_deg)\
                + f*exp(un) * un * vh * dx(mesh, degree = quad_deg)
        # Weingarten equation
        elif weingarten == True:
            A = ( H00 * phi00 + uh.dx(0) * phi00.dx(0) )*dx(mesh, degree = quad_deg)\
                + ( H11 * phi11 + uh.dx(1) * phi11.dx(1) )*dx(mesh, degree = quad_deg)\
                + ( H01 * phi01 + uh.dx(0) * phi01.dx(1) )*dx(mesh, degree = quad_deg)\
                - (phi00 * n[0]) * (uh.dx(0)) * ds(mesh, degree = quad_deg)\
                - (phi11 * n[1]) * (uh.dx(1)) * ds(mesh, degree = quad_deg)\
                - (phi01 * n[1]) * (uh.dx(0)) * ds(mesh, degree = quad_deg)\
                + (4.0*inner(gradn,graduh)*det(Hessn)/pow(1+pow(Dx,2)+pow(Dy,2),3)) * vh * dx(mesh, degree = quad_deg)\
                - ((Dyy*H00+Dxx*H11-2*Dxy*H01)/pow(1+pow(Dx,2)+pow(Dy,2),2)) * vh * dx(mesh, degree = quad_deg)\
                - ((H00+H11)/pow(1+pow(Dx,2)+pow(Dy,2),0.5)) * vh * dx(mesh, degree = quad_deg)\
                + ((pow(Dx,2)*H00+pow(Dy,2)*H11+2*Dx*Dy*H01)/pow(1+pow(Dx,2)+pow(Dy,2),1.5)) * vh * dx(mesh, degree = quad_deg)\
                + (inner(gradn,graduh)*(Dxx+Dyy)/pow(1+pow(Dx,2)+pow(Dy,2),1.5)) * vh * dx(mesh, degree = quad_deg)\
                + 2.0*((Dx*Dxx*uh.dx(0)+Dxy*(Dx*uh.dx(1)+Dy*uh.dx(0))+Dy*Dyy*uh.dx(1))/pow(1+pow(Dx,2)+pow(Dy,2),1.5)) * vh * dx(mesh, degree = quad_deg)\
                - (3.0*inner(gradn,graduh)/pow(1+pow(Dx,2)+pow(Dy,2),2.5))*(Dx*Dxx*Dx+Dxy*(Dx*Dy+Dy*Dx)+Dy*Dyy*Dy) * vh * dx(mesh, degree = quad_deg)
            L = - (f-det(Hessn)/pow(1+pow(Dx,2)+pow(Dy,2),2)) * vh * dx(mesh, degree = quad_deg)\
                + (((1+dot(gradn,gradn))*(Dxx+Dyy)-pow(Dx,2)*Dxx-2*Dx*Dy*Dxy-pow(Dy,2)*Dyy)/pow(1+pow(Dx,2)+pow(Dy,2),1.5)) * vh * dx(mesh, degree = quad_deg)\
                + (4.0*dot(gradn,gradn)*det(Hessn)/pow(1+pow(Dx,2)+pow(Dy,2),3)) * vh * dx(mesh, degree = quad_deg)\
                - ((Dyy*Dxx+Dxx*Dyy-2*Dxy*Dxy)/pow(1+pow(Dx,2)+pow(Dy,2),2)) * vh * dx(mesh, degree = quad_deg)\
                - ((Dxx+Dyy)/pow(1+pow(Dx,2)+pow(Dy,2),0.5)) * vh * dx(mesh, degree = quad_deg)\
                + ((pow(Dx,2)*Dxx+pow(Dy,2)*Dyy+2*Dx*Dy*Dxy)/pow(1+pow(Dx,2)+pow(Dy,2),1.5)) * vh * dx(mesh, degree = quad_deg)\
                + (dot(gradn,gradn)*(Dxx+Dyy)/pow(1+pow(Dx,2)+pow(Dy,2),1.5)) * vh * dx(mesh, degree = quad_deg)\
                + 2.0*((Dx*Dxx*Dx+Dxy*(Dx*Dy+Dy*Dx)+Dy*Dyy*Dy)/pow(1+pow(Dx,2)+pow(Dy,2),1.5)) * vh * dx(mesh, degree = quad_deg)\
                - (3.0*inner(gradn,gradn)/pow(1+pow(Dx,2)+pow(Dy,2),2.5))*(Dx*Dxx*Dx+Dxy*(Dx*Dy+Dy*Dx)+Dy*Dyy*Dy)* vh * dx(mesh, degree = quad_deg)
        # MA equation with RHS independent of u and grad(u)
        else:
            A = ( H00 * phi00 + uh.dx(0) * phi00.dx(0) )*dx(mesh, degree = quad_deg)\
                    + ( H11 * phi11 + uh.dx(1) * phi11.dx(1) )*dx(mesh, degree = quad_deg)\
                    + ( H01 * phi01 + uh.dx(0) * phi01.dx(1) )*dx(mesh, degree = quad_deg)\
                    - (phi00 * n[0]) * (uh.dx(0)) * ds(mesh, degree = quad_deg)\
                    - (phi11 * n[1]) * (uh.dx(1)) * ds(mesh, degree = quad_deg)\
                    - (phi01 * n[1]) * (uh.dx(0)) * ds(mesh, degree = quad_deg)\
                    - (Dyy*H00+Dxx*H11-2*Dxy*H01) * vh * dx(mesh, degree = quad_deg)
            L = -(f+Dxx*Dyy-Dxy*Dxy) * vh * dx(mesh, degree = quad_deg)
        

        U = Function(S)
        # begin timing of linear system solve
        if nitsche == True:
            A += + (10.0/hmin)*(uh)*vh*ds(mesh, degree = quad_deg)
            L += + (10.0/hmin)*(g)*vh*ds(mesh, degree = quad_deg)
            ttt = time()
            solve(A == L,U,
                  solver_parameters = {
                  "snes_type": "newtonls",
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "snes_monitor": False,
                  "snes_rtol": 1e-30,
                  "snes_atol": 1e-39})
            tt = (time()-ttt)
        else:
            ttt = time()
            solve(A == L,U, bcs = bc0,
                  solver_parameters = {
                  "snes_type": "newtonls",
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "snes_monitor": False,
                  "snes_rtol": 1e-30,
                  "snes_atol": 1e-39})
            tt = (time()-ttt)
        

        #splitting up solution for use in updating initial guesses.
        uh, H00, H11, H01 = U

        # Compute the number of degrees of freedom on the current mesh. For our mixed method it's dim(V) + d(d+1)*dim(W)/2
        ndof = (U.vector().array().size)
        L2relerr = sqrt(assemble((uh - uinit)**2*dx(mesh, degree = quad_deg)))

        return uh, H00, H11, H01, uh.dx(0), uh.dx(1), L2relerr, ndof, tt
    i = 0
    L2relerr = 1
    uinit = r*(0.5*(x**2+y**2-1))
    Dxuinit = r*x
    Dyuinit = r*y
    Dxxuinit = r*Constant(1.0)
    Dyyuinit = r*Constant(1.0)
    Dxyuinit = r*Constant(0.0)
    uinit0 = r*(0.5*(x**2+y**2-1))
    Dxuinit0 = r*x
    Dyuinit0 = r*y
    Dxxuinit0 = r*Constant(1.0)
    Dyyuinit0 = r*Constant(1.0)
    Dxyuinit0 = r*Constant(0.0)
    ndofs = 0
    initialdist = [];
    newtoncount = [];
    newtonerr = [];
    newtontimes = [];
    # Begin timing Newton's method
    ttt = time()
    while i < newtonitermax and L2relerr > newtontol:
        uinit, Dxxuinit, Dyyuinit, Dxyuinit, Dxuinit, Dyuinit, L2relerr, ndof, t = Model(Dxxuinit, Dxyuinit, Dyyuinit, Dxuinit, Dyuinit, uinit)
        print("L2 increment error:",L2relerr)
        newtoncount.append(i)
        newtonerr.append(L2relerr)
        newtontimes.append(t)
        i += 1
    # End timing Newton's method
    tt = time()-ttt
    if r>0:
        e_L2 = (sqrt(assemble((pow(uinit-u,2.0))*dx(mesh, degree = quad_deg))))
        e_H1 = (sqrt(assemble((pow(uinit.dx(0)-du0,2.0)+pow(uinit.dx(1)-du1,2.0))*dx(mesh, degree = quad_deg))))
        e_H2 = (sqrt(assemble(((pow(Dxxuinit-d2udxx,2.0)+pow(Dxyuinit-d2udxy,2.0)+pow(Dxyuinit-d2udxy,2.0)+pow(Dyyuinit-d2udyy,2.0))*dx(mesh, degree = quad_deg)))))
        e_H2t = (sqrt(assemble(((pow(uinit.dx(0).dx(0)-d2udxx,2.0)+pow(uinit.dx(0).dx(1)-d2udxy,2.0)+pow(uinit.dx(1).dx(0)-d2udxy,2.0)+pow(uinit.dx(1).dx(1)-d2udyy,2.0))*dx(mesh, degree = quad_deg)))))
    else:
        e_L2 = (sqrt(assemble((pow(uinit-(-u),2.0))*dx(mesh, degree = quad_deg))))
        e_H1 = (sqrt(assemble((pow(uinit.dx(0)-(-du0),2.0)+pow(uinit.dx(1)-(-du1),2.0))*dx(mesh, degree = quad_deg))))
        e_H2 = (sqrt(assemble(((pow(Dxxuinit-(-d2udxx),2.0)+pow(Dxyuinit-(-d2udxy),2.0)+pow(Dxyuinit-(-d2udxy),2.0)+pow(Dyyuinit-(-d2udyy),2.0))*dx(mesh, degree = quad_deg)))))
        e_H2t = (sqrt(assemble(((pow(uinit.dx(0).dx(0)-(-d2udxx),2.0)+pow(uinit.dx(0).dx(1)-(-d2udxy),2.0)+pow(uinit.dx(1).dx(0)-(-d2udxy),2.0)+pow(uinit.dx(1).dx(1)-(-d2udyy),2.0))*dx(mesh, degree = quad_deg)))))

    initialdist.append(sqrt(assemble((pow(uinit-uinit0,2)+pow(Dxxuinit-Dxxuinit0,2)+2.0*pow(Dxyuinit-Dxyuinit0,2)+pow(Dyyuinit-Dyyuinit0,2))*dx(mesh, degree = quad_deg))))

    return newtoncount, newtonerr, newtontimes, e_L2, e_H1, e_H2, e_H2t, h, ndof, tt, initialdist
for deg in [2,3]:
    e_L2 = []; e_H1 = []; e_H2 = []; e_H2t = []; hm = []; ndofs = []; dists = []; ntotlist = []; EOCL2 = []; EOCH1 = []; EOCH2 = []; EOCH2t = []; tt = []; ntotlist = [];
    EOCL2.append(0); EOCH1.append(0); EOCH2.append(0); EOCH2t.append(0)
    for idx in range(1,refinementsno):
        if deg == 2:
            dmp = 1.0
        else:
            dmp = 1.0
        newtoncount, newtonerr, newtontimes, e_L21, e_H11, e_H21, e_H2t1, hm1, ndofs1, tt1, indist = Newton(idx,deg,r)
        ntot = [];
        e_L2.append(e_L21); e_H1.append(e_H11); e_H2.append(e_H21); e_H2t.append(e_H2t1); hm.append(hm1); ndofs.append(ndofs1); tt.append(tt1); dists.append(indist); ntot.append(newtoncount[len(newtoncount)-1]); ntotlist.append(newtoncount[len(newtoncount)-1]);
        # Saving Newton iteration data to file
        if r>0:
            out_name11 = "MA-FiredrakeNewton-data/Newtonstepsrefinement_" + str(idx) + "p_" + str(deg) + "_convex.txt"
            out_name12 = "MA-FiredrakeNewton-data/Newtonincrementerrorrefinement_" + str(idx) + "p_" +str(deg)+ "_convex.txt"
            out_name13 = "MA-FiredrakeNewton-data/Newtontimesrefinement_" + str(idx) + "p_" +str(deg)+ "_convex.txt"
            out_name16 = "MA-FiredrakeNewton-data/Newtoninitdistrefinement_" + str(idx) + "p_" +str(deg)+ "_convex.txt"
            out_name17 = "MA-FiredrakeNewton-data/NewtonTotalstepsrefinement_" + str(idx) + "p_" + str(deg) + "_convex.txt"
        out_name11 = "MA-FiredrakeNewton-data/Newtonstepsrefinement_" + str(idx) + "p_" + str(deg) + "_concave.txt"
        out_name12 = "MA-FiredrakeNewton-data/Newtonincrementerrorrefinement_" + str(idx) + "p_" +str(deg)+ "_concave.txt"
        out_name13 = "MA-FiredrakeNewton-data/Newtontimesrefinement_" + str(idx) + "p_" +str(deg)+ "_concave.txt"
        out_name16 = "MA-FiredrakeNewton-data/Newtoninitdistrefinement_" + str(idx) + "p_" +str(deg)+ "_concave.txt"
        out_name17 = "MA-FiredrakeNewton-data/NewtonTotalstepsrefinement_" + str(idx) + "p_" + str(deg) + "_concave.txt"
        if dataoutput == True:
            np.savetxt(out_name11,newtoncount,fmt = '%s')
            np.savetxt(out_name12,newtonerr,fmt = '%s')
            np.savetxt(out_name13,newtontimes,fmt = '%s')
            np.savetxt(out_name16,indist,fmt = '%s')
            np.savetxt(out_name17,ntot,fmt = '%s')
        print("Newton steps:",newtoncount[len(newtoncount)-1])
        print("Increment errors:",newtonerr[len(newtonerr)-1])
    for k in range(1,len(e_L2)):
        EOCL2.append(ln(e_L2[k-1]/(e_L2[k]))/ln(hm[k-1]/hm[k]))
        EOCH1.append(ln(e_H1[k-1]/(e_H1[k]))/ln(hm[k-1]/hm[k]))
        EOCH2.append(ln(e_H2[k-1]/(e_H2[k]))/ln(hm[k-1]/hm[k]))
        EOCH2t.append(ln(e_H2t[k-1]/(e_H2t[k]))/ln(hm[k-1]/hm[k]))
    k = 0
    for k in range(len(e_L2)):
        print("Number of DOFs = ", ndofs[k])
        print("||u - u_h||_0 = ", e_L2[k], "   EOC = ", EOCL2[k])
        print("||u - u_h||_1 = ", e_H1[k], "   EOC = ", EOCH1[k])
        print("||u - u_h||_2 = ", e_H2[k], "   EOC = ", EOCH2[k])
        print("||u - u_h||_2,t = ", e_H2t[k], "   EOC = ", EOCH2t[k])
        k = k+1
    if r>0:
        out_name1 = "MA-FiredrakeNewton-data/e_L2p%i_convex.txt" %deg
        out_name2 = "MA-FiredrakeNewton-data/e_H1p%i_convex.txt" %deg
        out_name3 = "MA-FiredrakeNewton-data/e_H2p%i_convex.txt" %deg
        out_name5 = "MA-FiredrakeNewton-data/EOCL2p%i_convex.txt" %deg
        out_name6 = "MA-FiredrakeNewton-data/EOCH1p%i_convex.txt" %deg
        out_name7 = "MA-FiredrakeNewton-data/EOCH2p%i_convex.txt" %deg
        out_name9 = "MA-FiredrakeNewton-data/dofsp%i_convex.txt" %deg
        out_name10 = "MA-FiredrakeNewton-data/timesp%i_convex.txt" %deg
    out_name1 = "MA-FiredrakeNewton-data/e_L2p%i_concave.txt" %deg
    out_name2 = "MA-FiredrakeNewton-data/e_H1p%i_concave.txt" %deg
    out_name3 = "MA-FiredrakeNewton-data/e_H2p%i_concave.txt" %deg
    out_name5 = "MA-FiredrakeNewton-data/EOCL2p%i_concave.txt" %deg
    out_name6 = "MA-FiredrakeNewton-data/EOCH1p%i_concave.txt" %deg
    out_name7 = "MA-FiredrakeNewton-data/EOCH2p%i_concave.txt" %deg
    out_name9 = "MA-FiredrakeNewton-data/dofsp%i_concave.txt" %deg
    out_name10 = "MA-FiredrakeNewton-data/timesp%i_concave.txt" %deg
    if dataoutput == True:
        np.savetxt(out_name1,e_L2,fmt = '%s')
        np.savetxt(out_name2,e_H1,fmt = '%s')
        np.savetxt(out_name3,e_H2,fmt = '%s')
        np.savetxt(out_name5,EOCL2,fmt = '%s')
        np.savetxt(out_name6,EOCH1,fmt = '%s')
        np.savetxt(out_name7,EOCH2,fmt = '%s')
        np.savetxt(out_name9,ndofs,fmt = '%s')
        np.savetxt(out_name10,tt,fmt = '%s')
out_name10 = "MA-FiredrakeNewton-data/meshsize.txt"
if dataoutput == True:
    np.savetxt(out_name10,hm,fmt = '%s')