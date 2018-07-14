from firedrake import *
#Citations.print_at_exit()
import numpy as np
from time import time
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Make mesh ghosted for evaluation of DG terms
parameters["ghost_mode"] = "shared_facet"

# Specifying function space polynomial degree, degree of integration rule, choice of benchmark solution an operator, number of mesh refinements, nature of boundary condition, choice of domain.
deg = 4
quad_deg = 8
prob = 0
meshrefs = 7
inhomogeneous = False
domain = 0

# Definining minimum function via conditional statements
def makemin(a,b):
    mkmin = conditional(le(a,b),a,0)+conditional(le(b,a),b,0)
    return mkmin
def model(mucon,etacon,quad_deg,meshrefs,cstab,deg):
    e_L2 = []; e_H1 = []; e_H2 = []; e_h1 = []; ndof = [];
    hm = []; muvec = []; etavec = []; e_inf = []; e_projerr = []; tt = [];
    for idx in range(meshrefs):
        if domain == 0:
            mesh = Mesh("quasiunifrefdisk_%i.msh" % idx)
            Curved = True
        elif domain == 1:
            mesh = Mesh("keyhole_%i.msh" % idx)
            Curved = True
        else:
            mesh = UnitSquareMesh(2**(idx+1),2**(idx+1))
            bdry_indicator = 0.0
            Curved = False
        # Implementing quadratic domain approximation if the domain has a curved boundary portion. Note, this currently applies to  the unit disk, and the "key-hole" shaped domain.
        if Curved == True:
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
        else:
            mesh = UnitSquareMesh(2**(idx+1),2**(idx+1))
            bdry_indicator = 0.0
        # unit normal
        n = FacetNormal(mesh)

        # finite element space
        FES = FunctionSpace(mesh,"DG",deg)

        # Functions for defining bilinear form
        U = TrialFunction(FES)
        v = TestFunction(FES)
        theta = 0.5

        # defining maximum mesh size
        DG = FunctionSpace(mesh, "DG", 0);
        h = 0.5*(sqrt(max(interpolate(CellVolume(mesh), DG).vector()[:]))+sqrt(min(interpolate(CellVolume(mesh), DG).vector()[:])))

        # defining local mesh size
        hfct = sqrt(CellVolume(mesh))

        # defining jump stabilisation parameters
        muinner = cstab*mucon*pow(makemin(hfct('+'),hfct('-')),-1)
        muouter = cstab*mucon*pow(hfct,-1)
        etainner = cstab*etacon*pow(makemin(hfct('+'),hfct('-')),-3)
        etaouter = cstab*etacon*pow(hfct,-3)
        x, y = SpatialCoordinate(mesh)

        # defining true solution, coefficient matrix A, and boundary condition function for benchmarks
        if prob == 0:
            rho = pow(x,2)+pow(y,2)
            d2udyy = Constant(1.0)
            u = 0.25*sin(pi*rho)
            trueu = u
            du0 = 0.5*pi*x*cos(pi*rho)
            du1 = 0.5*pi*y*cos(pi*rho)
            d2udxx = 0.5*pi*cos(pi*rho)-pi**2*x**2*sin(pi*rho)
            d2udyy = 0.5*pi*cos(pi*rho)-pi**2*y**2*sin(pi*rho)
            d2udxy = -pi**2*x*y*sin(pi*rho)
            A = [[interpolate(Constant(1.0),FES),interpolate(Constant(0.0),FES)],[interpolate(Constant(0.0),FES),interpolate(Constant(1.0),FES)]]
            gamma = 1.0
            f = d2udxx+d2udyy
            g = u
            dg0 = du0
            dg1 = du1
            d2g00 = d2udxx
            d2g01 = d2udxy
            d2g11 = d2udyy
        
        elif prob == 1:
            alp = 0.5
            convup = 1.0
            rho = pow(pow(x,2)+pow(y,2),2)
            u = convup*(5.0*(rho-1.0)-alp*0.25*sin(pi*rho))
            du0 = convup*(10.0*x - alp*0.5*pi*x*cos(pi*rho))
            du1 = convup*(10.0*y - alp*0.5*pi*y*cos(pi*rho))
            d2udxx = convup*(Constant(10.0)-alp*0.5*pi*cos(pi*rho)+alp*pi**2*x**2*sin(pi*rho))
            d2udxy = convup*(Constant(0.0)+alp*pi**2*x*y*sin(pi*rho))
            d2udyy = convup*(Constant(10.0)-alp*0.5*pi*cos(pi*rho)+alp*pi**2*y**2*sin(pi*rho))
            f = 2.0*pow(d2udxx*d2udyy-pow(d2udxy,2.0),0.5)

            A = [[interpolate(Constant(1.0),FES),interpolate(Constant(0.0),FES)],[interpolate(Constant(0.0),FES),interpolate(Constant(1.0),FES)]]
            gamma = 1.0
            f = d2udxx + d2udyy
            g = u
            dg0 = du0
            dg1 = du1
            d2g00 = d2udxx
            d2g01 = d2udxy
            d2g11 = d2udyy
        elif prob == 2:
            rho = pow(x,2)+pow(y,2)
            u = 0.25*sin(pi*rho)
            du0 = 0.5*pi*x*cos(pi*rho)
            du1 = 0.5*pi*y*cos(pi*rho)
            d2udxx = 0.5*pi*cos(pi*rho)-pi**2*x**2*sin(pi*rho)
            d2udyy = 0.5*pi*cos(pi*rho)-pi**2*y**2*sin(pi*rho)
            d2udxy = -pi**2*x*y*sin(pi*rho)
            gamma = 2.0/5.0
            f = 2.0*(pi*cos(pi*rho)-pi**2*rho*sin(pi*rho))-2.0*pi**2*abs(x)*abs(y)*sin(pi*rho)
            A = [[Constant(2.0),(x*y)/(abs(x)*abs(y))],[(x*y)/(abs(x)*abs(y)),Constant(2.0)]]
            g = u
            dg0 = du0
            dg1 = du1
            d2g00 = d2udxx
            d2g01 = d2udxy
            d2g11 = d2udyy
        elif prob == 3:
            rho = pow(x,2)+pow(y,2)
            u = pow(rho,1.6)-1
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
            d2g00 = d2udxx
            d2g01 = d2udxy
            d2g11 = d2udyy
        elif prob == 4:
            rho = pow(x,2)+pow(y,2)
            u = pow(rho,0.8)-1
            du0 = 1.6*x*pow(rho,-0.2)
            du1 = 1.6*y*pow(rho,-0.2)
            d2udxx = 1.6*pow(rho,-0.2)-0.64*pow(x,2)*pow(rho,-1.2)
            d2udyy = 1.6*pow(rho,-0.2)-0.64*pow(y,2)*pow(rho,-1.2)
            d2udxy = -0.64*x*y*pow(rho,-1.2)
            f = 2*d2udxx+2*d2udyy+2*((x*y)/(abs(x)*abs(y)))*d2udxy
            A = [[Constant(2.0),(x*y)/(abs(x)*abs(y))],[(x*y)/(abs(x)*abs(y)),Constant(2.0)]]
            gamma = 2.0/5.0
            g = u
            dg0 = du0
            dg1 = du1
            d2g00 = d2udxx
            d2g01 = d2udxy
            d2g11 = d2udyy
        elif prob == 5:
            u = sin(pi*x)*sin(pi*y)
            du0 = pi*cos(pi*x)*sin(pi*y)
            du1 = pi*sin(pi*x)*cos(pi*y)
            d2udxx = -pi*pi*sin(pi*x)*sin(pi*y)
            d2udxy = pi*pi*cos(pi*x)*cos(pi*y)
            d2udyy = -pi*pi*sin(pi*x)*sin(pi*y)
            A = [[interpolate(Constant(1.0),FES),interpolate(Constant(0.0),FES)],[interpolate(Constant(0.0),FES),interpolate(Constant(1.0),FES)]]
            gamma = 1.0
            f = d2udxx+d2udyy
            g = u
            dg0 = du0
            dg1 = du1
            d2g00 = d2udxx
            d2g01 = d2udxy
            d2g11 = d2udyy
        # Tangential gradient operator
        def grad_T(u):
            gradT = as_vector([u.dx(0)-n[0]*(u.dx(0)*n[0]+u.dx(1)*n[1]),u.dx(1)-n[1]*(u.dx(0)*n[0]+u.dx(1)*n[1])])
            return gradT
        def B_star_1(u,v):
            Bstar1 = (u.dx(0).dx(0)*v.dx(0).dx(0)+u.dx(1).dx(0)*v.dx(1).dx(0)+u.dx(0).dx(1)*v.dx(0).dx(1)+u.dx(1).dx(1)*v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
            return Bstar1
        def extrabil(u,v,i,j):
            extra = (avg(u).dx(i)-(avg(u).dx(0)*n[0]('+')+avg(u).dx(1)*n[1]('+'))*n[i]('+'))*\
                (n[i]('+').dx(j)-(n[j]('+').dx(0)*n[0]('+')+n[j]('+').dx(1)*n[1]('+'))*n[i]('+'))*\
                (v.dx(j)('+')-v.dx(j)('-')-((v.dx(0)('+')-v.dx(0)('-'))*n[0]('+')+(v.dx(1)('+')-v.dx(1)('-'))*n[1]('+'))*n[j]('+'))
            
            return extra
        def extrabilpre(u,v):
            extra = extrabil(u,v,0,0)+extrabil(u,v,0,1)+extrabil(u,v,1,0)+extrabil(u,v,1,1)\
                    +(n[0]('+').dx(0)+n[1]('+').dx(1)+n[0]('+')*(n[0]('+').dx(0)*n[0]('+')+n[0]('+').dx(1)*n[1]('+'))+n[1]('+')*(n[1]('+').dx(0)*n[0]('+')+n[1]('+').dx(1)*n[1]('+')))\
                        *(avg(u).dx(0)*n[0]('+')+avg(u).dx(1)*n[1]('+'))*((v.dx(0)('+')-v.dx(0)('-'))*n[0]('+')+(v.dx(1)('+')-v.dx(1)('-'))*n[1]('+'))
            return extra
        def CurvedExtraTerm(u,v):
            extra = (extrabilpre(u,v)+extrabilpre(v,u))*dS(mesh,degree = quad_deg)+bdry_indicator*(u.dx(0)*n[0]+u.dx(1)*n[1])*(v.dx(0)*n[0]+v.dx(1)*n[1])*ds(mesh,degree = quad_deg)
            return extra
        
        
        
        
        #### internal face sum < div_Tavg(u) , jump(dv/dn) >
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
                        *dS(mesh,degree = quad_deg)
            return Bstar2a

    #### internal face sum < div_Tavg(v) , jump(du/dn) >

        def B_star_2_b(u,v):
            Bstar2b = (n[0]('+')*(u.dx(0)('+')-u.dx(0)('-'))+\
                       n[1]('+')*(u.dx(1)('+')-u.dx(1)('-'))\
                       ) \
                * \
                    (avg(v).dx(0).dx(0)+avg(v).dx(1).dx(1) \
                     -avg(v).dx(0).dx(0)*n[0]('+')*n[0]('+')\
                     -avg(v).dx(1).dx(0)*n[1]('+')*n[0]('+')\
                     -avg(v).dx(0).dx(1)*n[0]('+')*n[1]('+')\
                     -avg(v).dx(1).dx(1)*n[1]('+')*n[1]('+')\
                     ) \
                        *dS(mesh,degree = quad_deg)
            return Bstar2b
                
                   #### internal face sum <nabla_Tavg(du/dn) , jump(nabla_Tv)
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
                *dS(mesh,degree = quad_deg)
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
                        *dS(mesh,degree = quad_deg)
                
            return Bstar3b
                
                
                     #### external face sum < nabla_Tavg(du/dn) , jump(nabla_Tv) >
                     
        def B_star_4_a(u,v):
            Bstar4a =   (\
                         ((n[0]*u.dx(0)+n[1]*u.dx(1)).dx(0)-n[0]*n[0]*(n[0]*u.dx(0)+n[1]*u.dx(1)).dx(0) \
                          -n[0]*n[1]*(n[0]*u.dx(0)+n[1]*u.dx(1)).dx(1)\
                          ) \
                         * \
                         (v.dx(0)-n[0]*(n[0]*v.dx(0)+n[1]*v.dx(1)))\
                         + \
                         ((n[0]*u.dx(0)+n[1]*u.dx(1)).dx(1)-n[1]*n[0]*(n[0]*u.dx(0)+n[1]*u.dx(1)).dx(0) \
                          -n[1]*n[1]*(n[0]*u.dx(0)+n[1]*u.dx(1)).dx(1)\
                          ) \
                         * \
                         (v.dx(1)-n[1]*(n[0]*v.dx(0)+n[1]*v.dx(1)))\
                         )\
                        *ds(mesh,degree = quad_deg)
            return Bstar4a

        #### external face sum < nabla_Tavg(dv/dn) , jump(nabla_Tu) >
        def B_star_4_b(u,v):
            Bstar4b = (\
                   (u.dx(0)-n[0]*(n[0]*u.dx(0)+n[1]*u.dx(1)))\
                   * \
                   ((n[0]*v.dx(0)+n[1]*v.dx(1)).dx(0)-n[0]*n[0]*(n[0]*v.dx(0)+n[1]*v.dx(1)).dx(0) \
                    -n[0]*n[1]*(n[0]*v.dx(0)+n[1]*v.dx(1)).dx(1)\
                    ) \
                   + \
                   (u.dx(1)-n[1]*(n[0]*u.dx(0)+n[1]*u.dx(1)))\
                   * \
                   ((n[0]*v.dx(0)+n[1]*v.dx(1)).dx(1)-n[1]*n[0]*(n[0]*v.dx(0)+n[1]*v.dx(1)).dx(0) \
                    -n[1]*n[1]*(n[0]*v.dx(0)+n[1]*v.dx(1)).dx(1)\
                    ) \
                   )\
            * ds(mesh,degree = quad_deg)
            return Bstar4b

        # defining curvature dependent terms of the bilinear form
        def B_star(u,v):
            if Curved == True:
                Bstar = B_star_1(u,v)+B_star_2_a(u,v)+B_star_2_b(u,v)-B_star_3_a(u,v)-B_star_3_b(u,v)-B_star_4_a(u,v)-B_star_4_b(u,v)+CurvedExtraTerm(u,v)
            else:
                Bstar = B_star_1(u,v)+B_star_2_a(u,v)+B_star_2_b(u,v)-B_star_3_a(u,v)-B_star_3_b(u,v)-B_star_4_a(u,v)-B_star_4_b(u,v)
            return Bstar
        # Laplacian inner product
        def Deltainner(u,v):
            delt = (u.dx(0).dx(0)+u.dx(1).dx(1))*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh, degree = quad_deg)
            return delt
            
        # defining jump stabilisation operator
        def J_h_1(u,v):
            J1 = muinner*(n[0]('+')*(u.dx(0)('+')-u.dx(0)('-'))+n[1]('+')*(u.dx(1)('+')-u.dx(1)('-')))*(n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))+n[1]('+')*(v.dx(1)('+')-v.dx(1)('-')))*dS(mesh,degree = quad_deg)
            return J1
        def J_h_2_in_a(u,v):
            J2ina = muinner*(u.dx(0)('+')-u.dx(0)('-')-n[0]('+')*(n[0]('+')*(u.dx(0)('+')-u.dx(0)('-'))+n[1]('+')*(u.dx(1)('+')-u.dx(1)('-')))) \
                * (v.dx(0)('+')-v.dx(0)('-')-n[0]('+')*(n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))+n[1]('+')*(v.dx(1)('+')-v.dx(1)('-')))) * dS(mesh,degree = quad_deg)
            return J2ina
        def J_h_2_in_b(u,v):
            J2inb = muinner*(u.dx(1)('+')-u.dx(1)('-')-n[1]('+')*(n[0]('+')*(u.dx(0)('+')-u.dx(0)('-'))+n[1]('+')*(u.dx(1)('+')-u.dx(1)('-')))) \
                * (v.dx(1)('+')-v.dx(1)('-')-n[1]('+')*(n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))+n[1]('+')*(v.dx(1)('+')-v.dx(1)('-')))) * dS(mesh,degree = quad_deg)
            return J2inb
        def J_h_2_ext_a(u,v):
            J2exta = muouter*(u.dx(0)-n[0]*(u.dx(0)*n[0]+u.dx(1)*n[1]))*(v.dx(0)-n[0]*(v.dx(0)*n[0]+v.dx(1)*n[1]))*ds(mesh,degree = quad_deg)
            return J2exta
        def J_h_2_ext_b(u,v):
            J2extb = muouter*(u.dx(1)-n[1]*(u.dx(0)*n[0]+u.dx(1)*n[1]))*(v.dx(1)-n[1]*(v.dx(0)*n[0]+v.dx(1)*n[1]))*ds(mesh,degree = quad_deg)
            return J2extb
        def J_h_3_in(u,v):
            J3in = etainner*(u('+')-u('-'))*(v('+')-v('-'))*dS(mesh,degree = quad_deg)
            return J3in
        def J_h_3_ext(u,v):
            J3ext = etaouter*u*v*ds(mesh,degree = quad_deg)
            return J3ext
        
        def J_h(u,v):
            J = J_h_1(u,v) + J_h_2_in_a(u,v) + J_h_2_in_b(u,v) + J_h_2_ext_a(u,v) + J_h_2_ext_b(u,v) + J_h_3_in(u,v) + J_h_3_ext(u,v)
            return J

        # defining nondivergence part of the bilinear form
        def a(u,v):
            a = (A[0][0]*u.dx(0).dx(0)+A[1][1]*u.dx(1).dx(1)+A[1][0]*u.dx(1).dx(0)+A[0][1]*u.dx(0).dx(1))*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
            return a

        # defining bilinear form
        A_gamma = gamma*a(U,v)\
            +theta*B_star(U,v) \
            +(1-theta)*Deltainner(U,v)\
            +J_h(U,v)\
            -Deltainner(U,v)
        # defining linear form
        L = gamma*(f)*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh,degree = quad_deg)

        # altering the linear form for the inhomogeneous Dirichlet case
        if inhomogeneous == True:
            L= gamma*(f)*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh,degree = quad_deg)\
                +etaouter*g*v*ds(mesh,degree = quad_deg)\
                    +muouter*inner(as_vector([dg0-n[0]*(dg0*n[0]+dg1*n[1]),dg1-n[1]*(dg0*n[0]+dg1*n[1])]),grad_T(v))*ds(mesh,degree = quad_deg)\
                        - 0.5*(inner(grad_T(v.dx(0)*n[0]+v.dx(1)*n[1]),grad_T(g)))*ds(mesh,degree = quad_deg)\
                            - 0.5*((d2g00+d2g11-bdry_indicator*(dg0*n[0]+dg1*n[1])-(d2g00*n[0]*n[0]+2.0*d2g01*n[0]*n[1]+d2g11*n[1]*n[1])))*(v.dx(0)*n[0]+v.dx(1)*n[1])*ds(mesh,degree = quad_deg)
        # defining solution function
        U = Function(FES)

        # begin timing of linear system solve
        t = time()

        # solving linear system
        solve(A_gamma == L, U,
              solver_parameters = {
              "snes_type": "newtonls",
              "ksp_type": "preonly",
              "pc_type": "lu",
              "snes_monitor": False,
              "snes_rtol": 1e-16,
              "snes_atol": 1e-25})

        # end timing of linear system solve
        tt.append(time()-t)
        hm.append(h)

        # calculating errors
        errorL2 = (U-u)**2*dx(mesh,degree = quad_deg)
        errorH1 = ((U-u).dx(0)**2+(U-u).dx(1)**2)*dx(mesh,degree = quad_deg)
        errorH2 = ((U-u).dx(0).dx(0)**2+(U-u).dx(0).dx(1)**2+(U-u).dx(1).dx(0)**2+(U-u).dx(1).dx(1)**2)*dx(mesh,degree = quad_deg)
        if Curved == True:
            errorh1 = errorH2+J_h(U-u,U-u)+0.5*bdry_indicator*pow((U-u).dx(0)*n[0]+(U-u).dx(1)*n[1],2)*ds(mesh,degree = quad_deg)
        else:
            errorh1 = errorH2+J_h(U-u,U-u)
        eL2 = sqrt(assemble(errorL2))
        eH1 = sqrt(assemble(errorH1))
        eH2 = sqrt(assemble(errorH2))
        eh1 = sqrt(assemble(errorh1))
        e_L2.append(eL2)
        e_H1.append(eH1)
        e_H2.append(eH2)
        e_h1.append(eh1)

        # obtain number of DoFs
        ndof.append(U.vector().array().size)

    EOCL2 = []
    EOCH1 = []
    EOCH2 = []
    EOCh1 = []
    EOClinf = []
    EOCl2proj = []


    EOCL2.append(0)
    EOCH1.append(0)
    EOCH2.append(0)
    EOCh1.append(0)
    EOClinf.append(0)
    EOCl2proj.append(0)

    #Calcuating error orders of convergence.
    for k in range(1,len(e_L2)):
        EOCL2.append(ln(e_L2[k-1]/(e_L2[k]))/ln(hm[k-1]/hm[k]))
        EOCH1.append(ln(e_H1[k-1]/(e_H1[k]))/ln(hm[k-1]/hm[k]))
        EOCH2.append(ln(e_H2[k-1]/(e_H2[k]))/ln(hm[k-1]/hm[k]))
        EOCh1.append(ln(e_h1[k-1]/(e_h1[k]))/ln(hm[k-1]/hm[k]))

    k = 0

    # outputting jump stabilisation constants
    print("mucon = ",mucon)
    print("etacon = ",etacon)

    # outputting error results
    for k in range(len(e_L2)):
        print( "NDoFs = ", ndof[k])
        print( "Mesh size = ", hm[k])
        print( "run time = ",tt[k])
        print( "||u - u_h||_0 = ", e_L2[k], "   EOC = ", EOCL2[k])
        print( "||u - u_h||_1 = ", e_H1[k], "   EOC = ", EOCH1[k])
        print( "||u - u_h||_2 = ", e_H2[k], "   EOC = ", EOCH2[k])
        print( "||u - u_h||_h,1 = ", e_h1[k], "   EOC = ", EOCh1[k])
        k = k+1
    # returning errors, NDofs, runtimes, EOCs and polynomial degree for creation of data files
    return e_L2, e_H1, e_H2, e_h1, ndof, hm, tt,  EOCL2, EOCH1, EOCH2, EOCh1, deg
# solving problem for degrees 2,3 and 4, and saving data to text file
for degg in [2,3,4]:
    e_L2, e_H1, e_H2, e_h1, ndof, hm, tt,  EOCL2, EOCH1, EOCH2, EOCh1, deg = model(pow(deg,2)/2.0,3.0*pow(deg,4)/8.0,8,meshrefs,2.0,degg)
    print( "polynomial degree: ", deg)
    print( "integral degree: ", quad_deg)
    print( "experiment no. : ", prob)
    out_name1 = "e_L2p%i.txt" %deg
    out_name2 = "e_H1p%i.txt" %deg
    out_name3 = "e_H2p%i.txt" %deg
    out_name4 = "e_h-1p%i.txt" %deg
    out_name5 = "EOCL2p%i.txt" %deg
    out_name6 = "EOCH1p%i.txt" %deg
    out_name7 = "EOCH2p%i.txt" %deg
    out_name8 = "EOCh-1p%i.txt" %deg
    out_name9 = "dofsp%i.txt" %deg
    out_name10 = "meshsize.txt"
    out_name11 = "timesp%i.txt" %deg


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
    np.savetxt(out_name11,tt,fmt = '%s')
