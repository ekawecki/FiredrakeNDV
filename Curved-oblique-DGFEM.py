from firedrake import *
import numpy as np
from time import time
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Make mesh ghosted for evaluation of DG terms
parameters["ghost_mode"] = "shared_facet"

# Specifying function space polynomial degree, degree of integration rule, choice of benchmark solution and operator, choice of domain, number of mesh refinements, and whether or not we use a (precomputed) mesh of the unit disk that is refined towards the origin

deg = 4
quad_deg = 50
prob = 1
domain = 0
refinementsno = 6
meshref = False
if meshref == True:
    refinementsno = 2*refinementsno-2

# Definining minimum function via conditional statements
def makemin(a,b):
    mkmin = conditional(le(a,b),a,0)+conditional(le(b,a),b,0)
    return mkmin
def model(mucon,etacon,sigmacon,ellcon,quad_deg,meshrefs,cstab,deg):
    e_L2 = []; e_H1 = []; e_H2 = []; e_h1 = []; ndof = []; e_C = []; hm = []; muvec = []; etavec = []; e_inf = []; e_projerr = []; tt = []; hminm = [];
    for idx in range(refinementsno):
        if meshref == True:
            mesh = Mesh("Meshes/disk-reftocenter_%i.msh" % idx)
        else:
            mesh = Mesh("Meshes/quasiunifrefdisk_%i.msh" % idx)
        # Implementing quadratic domain approximation if the domain has a curved boundary portion. Note, this currently applies to the unit disk.
        V = FunctionSpace(mesh, "CG", 2)
        bdry_indicator = Function(V)
        bc = DirichletBC(V, Constant(1.0), 1)
        bc.apply(bdry_indicator)
        VV = VectorFunctionSpace(mesh, "CG", 2)
        T = Function(VV)
        T.interpolate(SpatialCoordinate(mesh))
        # Quadratic interpolation of the unit disk chart x/|x|
        T.interpolate(conditional(abs(1-bdry_indicator) < 1e-5, T/sqrt(inner(T,T)), T))
        mesh = Mesh(T)

        # finite element space, cross product of DG space with piecewise constant DG space
        FES = FunctionSpace(mesh,"DG",deg)
        Con1 = FunctionSpace(mesh,"DG",0)
        S = FES * Con1
        
        # Functions for defining bilinear form
        U, lamd = TrialFunction(S)
        v, mu = TestFunction(S)
        
        
        theta = 0.5
        
        # defining maximum and minimum mesh size
        DG = FunctionSpace(mesh, "DG", 0)
        h = sqrt(max(interpolate(CellVolume(mesh), DG).vector()[:]))
        hmin = sqrt(min(interpolate(CellVolume(mesh), DG).vector()[:]))
        
        # defining local mesh size
        hfct = sqrt(CellVolume(mesh))
        
        # defining jump stabilisation parameters
        muinner = cstab*mucon*pow(2*makemin(hfct('+'),hfct('-')),-1)
        muouter = cstab*mucon*pow(2*hfct,-1)
        etainner = cstab*etacon*pow(2*makemin(hfct('+'),hfct('-')),-3)
        etaouter = cstab*etacon*pow(2*hfct,-3)
        sigma = cstab*sigmacon*pow(2*hfct,-1)
        ellf = cstab*ellcon*pow(makemin(hfct('+'),hfct('-')),-3)
        x, y = SpatialCoordinate(mesh)
        
        #unit normal
        n = FacetNormal(mesh)
        
        # Tangential directional derivative
        def DT(u):
            dta = inner(Ta,grad(u))
            return dta
        
        # defining true solution, coefficient matrix A, and angle for oblique vector
        if prob == 0:
            rho = pow(x,2)+pow(y,2)
            u = 0.25*cos(pi*rho)
            u = u - assemble(u*dx(mesh,degree = quad_deg))/assemble(1.0*dx(mesh,degree = quad_deg))
            du0 = -0.5*pi*x*sin(pi*rho)
            du1 = -0.5*pi*y*sin(pi*rho)
            d2udxx = -0.5*pi*sin(pi*rho)-pow(pi,2)*pow(x,2)*cos(pi*rho)
            d2udyy = -0.5*pi*sin(pi*rho)-pow(pi,2)*pow(y,2)*cos(pi*rho)
            d2udxy = -pow(pi,2)*x*y*cos(pi*rho)
            Theta = pi/4.0+(1.0/16.0)*atan(y/x)
            oblvar = True
            truec = 0.0
            A = [[Constant(2.0),(x*y)/(abs(x)*abs(y))],[(x*y)/(abs(x)*abs(y)),Constant(2.0)]]
            gamma = 2.0/5.0
            f = A[0][0]*d2udxx+(A[0][1]+A[1][0])*d2udxy+A[1][1]*d2udyy
            g = u
            dg0 = du0
            dg1 = du1

        if prob == 1:
            rho = pow(x,2)+pow(y,2)
            u = (1.0/6.0)*pow(rho,3)-(1.0/2.0)*rho+5.0/24.0
            u = u-assemble(u*dx(mesh, degree = quad_deg))/assemble(1.0*dx(mesh, degree = quad_deg))
            du0 = x*pow(rho,2)+x
            du1 = y*pow(rho,2)+y
            d2udxx = (5.0*pow(x,2.0)+pow(y,2.0))*rho-1.0
            d2udxy = 4.0*x*y*rho
            d2udyy = (pow(x,2.0)+5.0*pow(y,2.0))*rho-1.0
            u = u - assemble(u*dx(mesh,degree = quad_deg))/assemble(1.0*dx(mesh,degree = quad_deg))
            
            A = [[interpolate(Constant(1.0),FES),interpolate(Constant(0.0),FES)],[interpolate(Constant(0.0),FES),interpolate(Constant(1.0),FES)]]
            gamma = 2.0
            f = A[0][0]*d2udxx+(A[0][1]+A[1][0])*d2udxy+A[1][1]*d2udyy
            Theta = 0.0
            oblvar = False
            truec = 0.0
            g = u
            dg0 = du0
            dg1 = du1
        elif prob == 2:
            rho = pow(x,2)+pow(y,2)
            u = pow(rho,0.8)-10.0/18.0
            du0 = x+0.5*x*pow(rho,-0.5)
            du1 = y+0.5*y*pow(rho,-0.5)
            d2udxx = 1.0+0.5*pow(y,2.0)*pow(rho,-3.0/2.0)
            d2udyy = 1.0+0.5*pow(x,2.0)*pow(rho,-3.0/2.0)
            d2udxy = -0.5*x*y*pow(rho,-3.0/2.0)
            A = [[interpolate(Constant(1.0),FES),interpolate(Constant(0.0),FES)],[interpolate(Constant(0.0),FES),interpolate(Constant(1.0),FES)]]
            gamma = 2.0
            f = d2udxx + d2udyy
            g = u
            dg0 = du0
            dg1 = du1
        elif prob == 3:
            rho = pow(x,2)+pow(y,2)
            u = (1.0/6.0)*pow(rho,3)-(1.0/2.0)*rho+5.0/24.0
            u = u-assemble(u*dx(mesh, degree = quad_deg))/assemble(1.0*dx(mesh, degree = quad_deg))
            du0 = x*pow(rho,2)-x
            du1 = y*pow(rho,2)-y
            d2udxx = (5.0*pow(x,2.0)+pow(y,2.0))*rho-1.0
            d2udxy = 4.0*x*y*rho
            d2udyy = (pow(x,2.0)+5.0*pow(y,2.0))*rho-1.0
            gamma = 2.0/5.0
            A = [[Constant(2.0),(x*y)/(abs(x)*abs(y))],[(x*y)/(abs(x)*abs(y)),Constant(2.0)]]
            f = Constant(2.0)*d2udxx+Constant(2.0)*d2udyy+2.0*((x*y)/(abs(x)*abs(y)))*d2udxy
            g = u
            dg0 = du0
            dg1 = du1
        elif prob == 4:
            rho = pow(x,2)+pow(y,2)
            u = pow(rho,1.6)-20.0/52.0
            u = u - assemble(u*dx(mesh,degree = quad_deg))/assemble(1.0*dx(mesh,degree = quad_deg))
            du0 = 1.6*2.0*x*pow(rho,0.6)
            du1 = 1.6*2.0*y*pow(rho,0.6)
            d2udxx = 3.84*x**2*pow(rho,-0.4)+3.2*pow(rho,0.6)
            d2udyy = 3.84*y**2*pow(rho,-0.4)+3.2*pow(rho,0.6)
            d2udxy = 3.84*pow(rho,-0.4)
            f = 7.68*pow(rho,0.6)+12.8*pow(rho,0.6)+7.68*abs(x)*abs(y)*pow(rho,-0.4)
            A = [[Constant(2.0),(x*y)/(abs(x)*abs(y))],[(x*y)/(abs(x)*abs(y)),Constant(2.0)]]
            gamma = 2.0/5.0
            g = u
            dg0 = du0
            dg1 = du1
        elif prob == 5:
            rho = pow(x,2)+pow(y,2)
            u = sin(0.5*rho)
            u = u-assemble(u*dx(mesh, degree = quad_deg))/assemble(1.0*dx(mesh, degree = quad_deg))
            du0 = x*cos(0.5*rho)
            du1 = y*cos(0.5*rho)
            d2udxx = cos(0.5*rho)-pow(x,2)*sin(0.5*rho)
            d2udxy = -x*y*sin(0.5*rho)
            d2udyy = cos(0.5*rho)-pow(y,2)*sin(0.5*rho)
            gamma = 2.0/5.0
            Theta = 0.0
            oblvar = False
            A = [[Constant(2.0),(x*y)/(abs(x)*abs(y))],[(x*y)/(abs(x)*abs(y)),Constant(2.0)]]
            f = Constant(2.0)*d2udxx+Constant(2.0)*d2udyy+2.0*((x*y)/(abs(x)*abs(y)))*d2udxy
            w = u
            g = u
            dg0 = du0
            dg1 = du1
        elif prob == 6:
            expon = 1.5/2.0
            rho = pow(x,2)+pow(y,2)
            u = pow(rho,expon)-expon*rho
            u = u - assemble(u*dx(mesh,degree = quad_deg))/assemble(1.0*dx(mesh,degree = quad_deg))
            du0 = 2.0*x*expon*pow(rho,expon-1)-2.0*expon*x
            du1 = 2.0*y*expon*pow(rho,expon-1)-2.0*expon*y
            d2udxx = 2.0*expon*pow(rho,expon-1)+4.0*pow(x,2)*expon*(expon-1)*pow(rho,expon-2)-2.0*expon
            d2udyy = 2.0*expon*pow(rho,expon-1)+4.0*pow(y,2)*expon*(expon-1)*pow(rho,expon-2)-2.0*expon
            d2udxy = 4.0*x*y*expon*(expon-1)*pow(rho,expon-2)
            A = [[Constant(2.0),(x*y)/(abs(x)*abs(y))],[(x*y)/(abs(x)*abs(y)),Constant(2.0)]]
            f = A[0][0]*d2udxx+(A[0][1]+A[1][0])*d2udxy+A[1][1]*d2udyy
            gamma = 2.0/5.0
            Theta = pi/4.0
            oblvar = False
            g = u
            dg0 = du0
            dg1 = du1
            truec = 0.0
        elif prob == 7:
            expon = 1.5/2.0
            rho = pow(x,2)+pow(y,2)
            u = pow(rho,expon)-expon*rho
            u = u - assemble(u*dx(mesh,degree = quad_deg))/assemble(1.0*dx(mesh,degree = quad_deg))
            du0 = 2.0*x*expon*pow(rho,expon-1)-2.0*expon*x
            du1 = 2.0*y*expon*pow(rho,expon-1)-2.0*expon*y
            d2udxx = 2.0*expon*pow(rho,expon-1)+4.0*pow(x,2)*expon*(expon-1)*pow(rho,expon-2)-2.0*expon
            d2udyy = 2.0*expon*pow(rho,expon-1)+4.0*pow(y,2)*expon*(expon-1)*pow(rho,expon-2)-2.0*expon
            d2udxy = 4.0*x*y*expon*(expon-1)*pow(rho,expon-2)
            A = [[interpolate(Constant(1.0),FES),interpolate(Constant(0.0),FES)],[interpolate(Constant(0.0),FES),interpolate(Constant(1.0),FES)]]
            f = 1
            gamma = 2.0
            Theta = 0.0+(2.0)*atan(y/x)
            oblvar = True
            g = u
            dg0 = du0
            dg1 = du1
        elif prob == 8:
            alp = 0.05
            convup = 1.0
            rho = pow(x,2)+pow(y,2)
            u = convup*(5.0*(rho-1.0)+alp*0.25*sin(pi*rho))
            u = u - assemble(u*dx(mesh,degree = quad_deg))/assemble(1.0*dx(mesh,degree = quad_deg))
            du0 = convup*(10.0*x + alp*0.5*pi*x*cos(pi*rho))
            du1 = convup*(10.0*y + alp*0.5*pi*y*cos(pi*rho))
            d2udxx = convup*(Constant(10.0)+alp*0.5*pi*cos(pi*rho)-alp*pi**2*x**2*sin(pi*rho))
            d2udxy = convup*(Constant(0.0)-alp*pi**2*x*y*sin(pi*rho))
            d2udyy = convup*(Constant(10.0)+alp*0.5*pi*cos(pi*rho)-alp*pi**2*y**2*sin(pi*rho))
            A = [[interpolate(Constant(1.0),FES),interpolate(Constant(0.0),FES)],[interpolate(Constant(0.0),FES),interpolate(Constant(1.0),FES)]]
            gamma = 2.0
            f = d2udxx + d2udyy
            gamma = 2.0
            Theta = 0.0
            oblvar = False
            g = u
            dg0 = du0
            dg1 = du1
            truec = 10.0-alp*0.5*pi
        
        # Defining oblique vector and oblique angle
        cc = cos(Theta)
        ss = sin(Theta)
        Ta = as_vector([-n[1],n[0]])
        beta = as_vector([cc*n[0]-ss*n[1],ss*n[0]+cc*n[1]])
        betaperp = as_vector([-beta[1],beta[0]])
        obliqueangle = acos(inner(beta,n))
        truech = inner(beta,as_vector([du0,du1]))
        
        # Tangential gradient
        def grad_T(u):
            gradT = as_vector([u.dx(0)-n[0]*(u.dx(0)*n[0]+u.dx(1)*n[1]),u.dx(1)-n[1]*(u.dx(0)*n[0]+u.dx(1)*n[1])])
            return gradT
        def grad_Touter(u):
            gradTout = as_vector([u.dx(0)-n[0]*(u.dx(0)*n[0]+u.dx(1)*n[1]),u.dx(1)-n[1]*(u.dx(0)*n[0]+u.dx(1)*n[1])])
            return gradTout

        
        def delta_T(u):
            delt = (u.dx(0).dx(0)+u.dx(1).dx(1)-u.dx(0)*n[0]-u.dx(1)*n[1]\
                    -u.dx(0).dx(0)*n[0]*n[0]\
                    -u.dx(1).dx(0)*n[1]*n[0]\
                    -u.dx(0).dx(1)*n[0]*n[1]\
                    -u.dx(1).dx(1)*n[1]*n[1]\
                    )
            return delt
        def delta_Touter(u):
            deltout = (u.dx(0).dx(0)+u.dx(1).dx(1)-u.dx(0)*n[0]-u.dx(1)*n[1]\
                -u.dx(0).dx(0)*n[0]*n[0]\
                -u.dx(1).dx(0)*n[1]*n[0]\
                -u.dx(0).dx(1)*n[0]*n[1]\
                -u.dx(1).dx(1)*n[1]*n[1]\
                )
            return deltout
        def B_star_1(u,v):
            Bstar1 = (u.dx(0).dx(0)*v.dx(0).dx(0)+u.dx(1).dx(0)*v.dx(1).dx(0)+u.dx(0).dx(1)*v.dx(0).dx(1)+u.dx(1).dx(1)*v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
            return Bstar1
        # Curvature and oblique angle dependent bilinear form terms
        if oblvar == True:
            def CurvedExtraTerm(u,v):
                extra = (DT(obliqueangle)+1.0)*inner(grad(u),grad(v))*ds(mesh,degree = quad_deg)
                return extra
        else:
            def CurvedExtraTerm(u,v):
                extra = inner(grad(u),grad(v))*ds(mesh,degree = quad_deg)
                return extra
        
        
        
        
        #### internal face sum < div_Tgrad_Tavg(u) , jump(dv/dn) >
        def B_star_2_a(u,v):
            Bstar2a = (avg(u).dx(0).dx(0)+avg(u).dx(1).dx(1) \
                       -avg(u).dx(0).dx(0)*n[0]('+')*n[0]('+')\
                       -avg(u).dx(1).dx(0)*n[1]('+')*n[0]('+')\
                       -avg(u).dx(0).dx(1)*n[0]('+')*n[1]('+')\
                       -avg(u).dx(1).dx(1)*n[1]('+')*n[1]('+')\
                       ) \
                * \
                    (n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))+\
                     n[1]('+')*(v.dx(1)('+')-v.dx(1)('-'))
                     )\
                        *dS(mesh, degree = quad_deg)
            return Bstar2a

        #### internal face sum <nabla_Tavg(du/dn) , jump(nabla_Tv) >
        def B_star_2_b(u,v):
            Bstar2b =   (\
                         (((n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(0) \
                           -n[0]('+')*n[0]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(0) \
                           -n[0]('+')*n[1]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(1) \
                           )) \
                         * \
                         ( \
                          v.dx(0)('+')-v.dx(0)('-')-n[0]('+')*n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))\
                          -n[0]('+')*n[1]('+')*(v.dx(1)('+')-v.dx(1)('-'))\
                          )\
                         +(\
                           ((n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(1) \
                            -n[1]('+')*n[0]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(0) \
                            -n[1]('+')*n[1]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(1) \
                            )) \
                         * \
                         ( \
                          v.dx(1)('+')-v.dx(1)('-')-n[1]('+')*n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))\
                          -n[1]('+')*n[1]('+')*(v.dx(1)('+')-v.dx(1)('-'))\
                          )\
                         )\
                *dS(mesh, degree = quad_deg)
            return Bstar2b
                
                
        #### external face sum < Delta_Tu , dv/dn >
                     
        def B_star_3(u,v,l):
            Bstar3 = DT(inner(betaperp,grad(u)))*(inner(beta,grad(v))-l)*ds(mesh,degree = quad_deg)
            return Bstar3

        #### internal face sum <nabla_Tavg(du/dn) , jump(nabla_Tv) >
        def B_star_3_a(u,v):
            Bstar3a =   (\
                         (((n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(0) \
                           -n[0]('+')*n[0]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(0) \
                           -n[0]('+')*n[1]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(1) \
                           )) \
                         * \
                         ( \
                          v.dx(0)('+')-v.dx(0)('-')-n[0]('+')*n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))\
                          -n[0]('+')*n[1]('+')*(v.dx(1)('+')-v.dx(1)('-'))\
                          )\
                         +(\
                           ((n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(1) \
                            -n[1]('+')*n[0]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(0) \
                            -n[1]('+')*n[1]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(1) \
                            )) \
                         * \
                         ( \
                          v.dx(1)('+')-v.dx(1)('-')-n[1]('+')*n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))\
                          -n[1]('+')*n[1]('+')*(v.dx(1)('+')-v.dx(1)('-'))\
                          )\
                         )\
            *dS(mesh, degree = quad_deg)
            return Bstar3a
                
        #### internal face sum < nabla_Tavg(dv/dn) , jump(nabla_Tu) >
        def B_star_3_b(u,v):
            Bstar3b =   (\
                         ( \
                          u.dx(0)('+')-u.dx(0)('-')-n[0]('+')*n[0]('+')*(u.dx(0)('+')-u.dx(0)('-'))\
                          -n[0]('+')*n[1]('+')*(u.dx(1)('+')-u.dx(1)('-'))\
                          )\
                         * \
                         (((n[0]('+')*avg(v.dx(0))+n[1]('+')*avg(v.dx(1))).dx(0) \
                           -n[0]('+')*n[0]('+')*(n[0]('+')*avg(v.dx(0))+n[1]('+')*avg(v.dx(1))).dx(0) \
                           -n[0]('+')*n[1]('+')*(n[0]('+')*avg(v.dx(0))+n[1]('+')*avg(v.dx(1))).dx(1) \
                           )) \
                         +
                         ( \
                          u.dx(1)('+')-u.dx(1)('-')-n[1]('+')*n[0]('+')*(u.dx(0)('+')-u.dx(0)('-'))\
                          -n[1]('+')*n[1]('+')*(u.dx(1)('+')-u.dx(1)('-'))\
                          )\
                         * \
                         (\
                          ((n[0]('+')*avg(v.dx(0))+n[1]('+')*avg(v.dx(1))).dx(1) \
                           -n[1]('+')*n[0]('+')*(n[0]('+')*avg(v.dx(0))+n[1]('+')*avg(v.dx(1))).dx(0) \
                           -n[1]('+')*n[1]('+')*(n[0]('+')*avg(v.dx(0))+n[1]('+')*avg(v.dx(1))).dx(1) \
                           )) \
                         ) \
                *dS(mesh, degree = quad_deg)
                    
            return Bstar3b


        #### external face sum < du/dn, Delta_Tv >
        def B_star_4(u,v):
            Bstar4 =(u.dx(0)*n[0]+u.dx(1)*n[1])\
                *(delta_Touter(v))\
                    *ds(mesh, degree = quad_deg)
            return Bstar4

        def B_star_6(u,v):
            Bstar6 = 0.5*(u.dx(0)*n[0]+u.dx(1)*n[1])\
                *(v.dx(0)*n[0]+v.dx(1)*n[1])\
                    * ds(mesh, degree = quad_deg)
            return Bstar6
        
        ### external face sum <grad_t(du/dn),grad_Tv>
        def B_star_7(u,v):
            Bstar7 = inner(grad_T(u.dx(0)*n[0]+u.dx(1)*n[1]),grad_T(v)) * ds(mesh, degree = quad_deg)
            return Bstar7
        def B_star_5(u,v):
            Bstar5 = DT(inner(beta,grad(u)))*inner(betaperp,grad(v))*ds(mesh, degree = quad_deg)
            return Bstar5

        def B_star(u,v,l,mu):
            Bstar = B_star_1(u,v)+B_star_2_a(u,v)+B_star_2_a(v,u)-B_star_2_b(u,v)-B_star_2_b(v,u)+B_star_3(u,v,mu)+B_star_3(v,u,l)+CurvedExtraTerm(u,v)
            return Bstar
        
        # Laplacian inner product
        def Deltainner(u,v):
            delt = (u.dx(0).dx(0)+u.dx(1).dx(1))*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh, degree = quad_deg)
            return delt


        # defining jump stabilisation operator
        def J_h_1(u,v):
            J1 = muinner*(n[0]('+')*(u.dx(0)('+')-u.dx(0)('-'))+n[1]('+')*(u.dx(1)('+')-u.dx(1)('-')))*(n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))+n[1]('+')*(v.dx(1)('+')-v.dx(1)('-')))*dS(mesh, degree = quad_deg)
            return J1
        def J_h_2_in_a(u,v):
            J2ina = muinner*(u.dx(0)('+')-u.dx(0)('-')-n[0]('+')*(n[0]('+')*(u.dx(0)('+')-u.dx(0)('-'))+n[1]('+')*(u.dx(1)('+')-u.dx(1)('-')))) \
                * (v.dx(0)('+')-v.dx(0)('-')-n[0]('+')*(n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))+n[1]('+')*(v.dx(1)('+')-v.dx(1)('-')))) * dS(mesh, degree = quad_deg)
            return J2ina
        def J_h_2_in_b(u,v):
            J2inb = muinner*(u.dx(1)('+')-u.dx(1)('-')-n[1]('+')*(n[0]('+')*(u.dx(0)('+')-u.dx(0)('-'))+n[1]('+')*(u.dx(1)('+')-u.dx(1)('-')))) \
            * (v.dx(1)('+')-v.dx(1)('-')-n[1]('+')*(n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))+n[1]('+')*(v.dx(1)('+')-v.dx(1)('-')))) * dS(mesh, degree = quad_deg)
            return J2inb
        def J_h_2_ext_a(u,v):
            J2exta = muouter*(u.dx(0)-n[0]*(u.dx(0)*n[0]+u.dx(1)*n[1]))*(v.dx(0)-n[0]*(v.dx(0)*n[0]+v.dx(1)*n[1]))*ds(mesh, degree = quad_deg)
            return J2exta
        def J_h_2_ext_b(u,v):
            J2extb = muouter*(u.dx(1)-n[1]*(u.dx(0)*n[0]+u.dx(1)*n[1]))*(v.dx(1)-n[1]*(v.dx(0)*n[0]+v.dx(1)*n[1]))*ds(mesh, degree = quad_deg)
            return J2extb
        def J_h_3_in(u,v):
            J3in = etainner*(u('+')-u('-'))*(v('+')-v('-'))*dS(mesh, degree = quad_deg)
            return J3in
        def J_h_3_ext(u,v,l,mu):
            J3ext = sigma*(inner(beta,grad(u))-l)*(inner(beta,grad(v))-mu)*ds(mesh, degree = quad_deg)
            return J3ext
            return J4ext
        def J_h_5_ext(u,v):
            J5ext = sigmajump*((inner(betaperp,grad(u)))*ds(mesh, degree = quad_deg))*((inner(betaperp,grad(v)))*ds(mesh, degree = quad_deg))
            return J5ext
        def J_h_fix(l,mu):
            fix = ellf*(l('+')-l('-'))*(mu('+')-mu('-'))*dS(mesh, degree = quad_deg)
            return fix
        
        def J_h(u,v,l,mu):
            J = J_h_1(u,v) + J_h_2_in_a(u,v) + J_h_2_in_b(u,v) + J_h_3_in(u,v) + J_h_3_ext(u,v,l,mu) +J_h_3_in(l,mu)+J_h_fix(l,mu)
            return J
        
        # defining nondivergence part of the bilinear form
        def a(u,v):
            a = (A[0][0]*u.dx(0).dx(0)+A[1][1]*u.dx(1).dx(1)+A[1][0]*u.dx(1).dx(0)+A[0][1]*u.dx(0).dx(1))*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
            return a
        
        # defining bilinear form
        A_gamma = gamma*a(U,v)\
            +theta*B_star(U,v,lamd,mu)\
            +(1-theta)*Deltainner(U,v)\
            +J_h(U,v,lamd,mu)\
            -Deltainner(U,v)
        
        # defining linear form
        L = gamma*(f)*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh, degree = quad_deg)
        Uh = Function(S)
        
        # implementing nullspace, as solution should have zero sum
        V_basis = VectorSpaceBasis(constant=True)
        nullspace = MixedVectorSpaceBasis(S, [V_basis, S[1]])

        # begin timing of linear system solve
        t = time()
        
        # solving linear system
        solve(A_gamma == L,Uh,nullspace = nullspace,
                  solver_parameters = {"mat_type": "aij",
                  "snes_type": "newtonls",
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "snes_monitor": False,
                  "snes_rtol": 1e-16,
                  "snes_atol": 1e-25})
        # end timing of linear system solve
        tt.append(time()-t)
        hm.append(h)
        hminm.append(hmin)
        
        #calculating errors
        uh, ch = Uh
        Uuavg = assemble((uh-u)*dx(mesh, degree = quad_deg))/assemble(1.0*dx(mesh, degree = quad_deg))
        errorL2 = (uh-u-Uuavg)**2*dx(mesh, degree = quad_deg)
        errorH1 = ((uh-u).dx(0)**2+(uh-u).dx(1)**2)*dx(mesh, degree = quad_deg)
        errorH2 = ((uh-u).dx(0).dx(0)**2+(uh-u).dx(0).dx(1)**2+(uh-u).dx(1).dx(0)**2+(uh-u).dx(1).dx(1)**2)*dx(mesh, degree = quad_deg)
        errorh1 = errorH2+cstab*J_h(uh-u,uh-u,ch-truec,ch-truec)+cstab*(pow((uh-u).dx(0),2)+pow((uh-u).dx(1),2))*ds(mesh, degree = quad_deg)
        
        eL2 = sqrt(assemble(errorL2))
        eH1 = sqrt(assemble(errorH1))
        eH2 = sqrt(assemble(errorH2))
        eh1 = sqrt(assemble(errorh1))
        eC = sqrt(assemble(pow(ch-truec,2)*ds(mesh,degree = quad_deg)))
        e_L2.append(eL2)
        e_H1.append(eH1)
        e_H2.append(eH2)
        e_h1.append(eh1)
        e_C.append(eC)
        
        # obtain number of DoFs
        ndof.append(Uh.vector().array().size)

    EOCL2 = []
    EOCH1 = []
    EOCH2 = []
    EOCh1 = []
    EOCC = []
    EOCh1ndofs = []
    EOChndof = []
    EOCH2ndofs = []
    EOCL2.append(0)
    EOCH1.append(0)
    EOCH2.append(0)
    EOCh1.append(0)
    EOCC.append(0)
    EOCh1ndofs.append(0)
    EOCH2ndofs.append(0)
    EOChndof.append(0)

    #Calcuating error orders of convergence.
    for k in range(1,len(e_L2)):
        EOCL2.append(ln(e_L2[k-1]/(e_L2[k]))/ln(hm[k-1]/hm[k]))
        EOCH1.append(ln(e_H1[k-1]/(e_H1[k]))/ln(hm[k-1]/hm[k]))
        EOCH2.append(ln(e_H2[k-1]/(e_H2[k]))/ln(hm[k-1]/hm[k]))
        EOCh1.append(ln(e_h1[k-1]/(e_h1[k]))/ln(hm[k-1]/hm[k]))
        EOCC.append(ln(e_C[k-1]/(e_C[k]))/ln(hm[k-1]/hm[k]))
        EOCh1ndofs.append(ln(e_h1[k-1]/(e_h1[k]))/ln(ndof[k-1]/ndof[k]))
        EOCH2ndofs.append(ln(e_H2[k-1]/(e_H2[k]))/ln(ndof[k-1]/ndof[k]))
        EOChndof.append(ln(hm[k-1]/(hm[k]))/ln(ndof[k-1]/ndof[k]))


    k = 0
    for k in range(len(e_L2)):
        print( "NDoFs = ", ndof[k])
        print( "Mesh size = ", hm[k])
        print( "run time = ",tt[k])
        print("||u - u_h||_h,1 = ",e_h1[k],"||.||_h,1 ndof EOC = ", EOCh1ndofs[k])
        print("h ndof EOC = ", EOChndof[k])
        print("||u - u_h||_2 = ",e_H2[k], "|.|_H2 ndof EOC = ", EOCH2ndofs[k])
        print("coarse h = ",hm[k],"fine h = ",hminm[k],"f/c ratio = ",hminm[k]/hm[k])
        print("||u - u_h||_0 = ", e_L2[k], "   EOC = ", EOCL2[k])
        print("||u - u_h||_1 = ", e_H1[k], "   EOC = ", EOCH1[k])
        print("||u - u_h||_2 = ", e_H2[k], "   EOC = ", EOCH2[k])
        print("||u - u_h||_h,1 = ", e_h1[k], "   EOC = ", EOCh1[k])
        print("||c - c_h|| = ", e_C[k], "   EOC = ", EOCC[k])
        k = k+1
    # returning errors, NDofs, runtimes, EOCs and polynomial degree for creation of data files
    return e_L2, e_H1, e_H2, e_h1, ndof, hm, tt,  EOCL2, EOCH1, EOCH2, EOCh1, e_C, EOCC, EOChndof, deg

# solving problem for degrees 2, 3 and 4, and saving data to text file
for degg in [2,3,4]:
    e_L2, e_H1, e_H2, e_h1, ndof, hm, tt,  EOCL2, EOCH1, EOCH2, EOCh1, e_C, EOCC, EOCh1ndofs, deg = model(pow(degg-1,2),0.15625*6.0*pow(degg-1,4)/8.0,1.0,1.0,50,refinementsno,2.5,degg)
    print("polynomial degree: ", deg)
    print("experiment no. : ", prob)
    out_name1 = "Curved-oblique-DGFEM-data/e_L2p%i.txt" %deg
    out_name2 = "Curved-oblique-DGFEM-data/e_H1p%i.txt" %deg
    out_name5 = "Curved-oblique-DGFEM-data/EOCL2p%i.txt" %deg
    out_name6 = "Curved-oblique-DGFEM-data/EOCH1p%i.txt" %deg
    out_name7 = "Curved-oblique-DGFEM-data/EOCH2p%i.txt" %deg
    out_name8 = "Curved-oblique-DGFEM-data/EOCh-1p%i.txt" %deg
    if meshref == True:
        out_name9 = "Curved-oblique-DGFEM-data/rdofsp%i.txt" %deg
        out_name3 = "Curved-oblique-DGFEM-data/re_H2p%i.txt" %deg
        out_name4 = "Curved-oblique-DGFEM-data/re_h-1p%i.txt" %deg
    else:
        out_name9 = "Curved-oblique-DGFEM-data/dofsp%i.txt" %deg
        out_name3 = "Curved-oblique-DGFEM-data/e_H2p%i.txt" %deg
        out_name4 = "Curved-oblique-DGFEM-data/e_h-1p%i.txt" %deg
    out_name10 = "Curved-oblique-DGFEM-data/meshsize.txt"
    out_name11 = "Curved-oblique-DGFEM-data/e_Cp%i.txt" %deg
    out_name12 = "Curved-oblique-DGFEM-data/EOCCp%i.txt" %deg

    out_name13 = "Curved-oblique-DGFEM-data/ndofEOCh-1p%i.txt" %deg

    text_file = open("Curved-oblique-DGFEM-data/exp_%i_info.txt" %prob,"w")
    text_file.write("experiment number: %s\n" % prob)
    text_file.write("polynomial degree: %s\n" % deg)
    text_file.write("L2 error: %s\n" % e_L2)
    text_file.write("L2 eoc: %s\n" % EOCL2)
    text_file.write("H1 error: %s\n" % e_H1)
    text_file.write("H1 eoc: %s\n" % EOCH1)
    text_file.write("H2 error: %s\n" % e_H2)
    text_file.write("H2 eoc: %s\n" % EOCH2)
    text_file.write("h1 error: %s\n" % e_h1)
    text_file.write("h1 eoc: %s\n" % EOCh1)
    text_file.write("C error: %s\n" % e_C)
    text_file.write("C eoc: %s\n" % EOCC)
    text_file.write("meshsizes: %s\n" % hm)
    text_file.write("ndofs: %s\n" % ndof)

    np.savetxt(out_name1,e_L2,fmt = '%s')
    np.savetxt(out_name2,e_H1,fmt = '%s')
    np.savetxt(out_name3,e_H2,fmt = '%s')
    np.savetxt(out_name4,e_h1,fmt = '%s')
    np.savetxt(out_name5,EOCL2,fmt = '%s')
    np.savetxt(out_name6,EOCH1,fmt = '%s')
    np.savetxt(out_name7,EOCH2,fmt = '%s')
    np.savetxt(out_name8,EOCh1,fmt = '%s')
    np.savetxt(out_name9,ndof,fmt = '%s')
    np.savetxt(out_name10,hm,fmt = '%s')
    np.savetxt(out_name11,e_C,fmt = '%s')
    np.savetxt(out_name12,EOCC,fmt = '%s')
    np.savetxt(out_name13,EOCh1ndofs,fmt = '%s')
