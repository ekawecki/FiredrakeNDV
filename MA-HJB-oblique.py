from firedrake import *
import numpy as np
from time import time

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Make mesh ghosted for evaluation of DG terms
parameters["ghost_mode"] = "shared_facet"

# Specifying function space polynomial degree, degree of integration rule, choice of benchmark solution and problem (HJB or MA), number of mesh refinements, maximum Newton method iterations, Newton method tolerance, choice of domain, and nature of boundary condition.
deg = 2
quad_deg = 50
prob = 1
refinementsno = 7
newtonitermax = 20
newtontol = 1e-12
disk = True
cstab = 2.5
set_log_level(1)
# Defining min, max and sign functions via conditional statements
def makemax(a,b):
    mkmax = conditional(ge(a,b),a,0)+conditional(ge(b,a),b,0)
    return mkmax
def signfct(a):
    sgn = conditional(ge(a,0),1,0)+conditional(ge(0,a),-1,0)
    return sgn
def makemin(a,b):
    mkmin = conditional(le(a,b),a,0)+conditional(le(b,a),b,0)
    return mkmin

def Newton(idx,mucon,etacon,sigmacon,ellcon,quad_deg,cstab):
    if disk == True:
        mesh = Mesh("Meshes/quasiunifrefdisk_%i.msh" % idx)
    else:
        UnitSquareMesh(2**(idx+1),2**(idx+1))
        bdry_indicator = 0.0
    V = FunctionSpace(mesh, "CG", 2)

    # Implementing quadratic domain approximation if the domain has a curved boundary portion.
    if disk == True:
        bdry_indicator = Function(V)
        bc = DirichletBC(V, Constant(1.0), 1)
        bc.apply(bdry_indicator)
        VV = VectorFunctionSpace(mesh, "CG", 2)
        T = Function(VV)
        T.interpolate(SpatialCoordinate(mesh))
        # Quadratic interpolation of the unit disk chart x/|x|
        T.interpolate(conditional(abs(1-bdry_indicator) < 1e-5, T/sqrt(inner(T,T)), T))
        mesh = Mesh(T)
    
    # unit normal
    n = FacetNormal(mesh)


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



    # defining benchmark solutions and nonlinear problems
    if prob == 0:
        rho = pow(x,2)+pow(y,2)
        u = 0.25*cos(pi*rho)
        du0 = -0.5*pi*x*sin(pi*rho)
        du1 = -0.5*pi*y*sin(pi*rho)
        d2udxx = -0.5*pi*sin(pi*rho)-pow(pi,2)*pow(x,2)*cos(pi*rho)
        d2udyy = -0.5*pi*sin(pi*rho)-pow(pi,2)*pow(y,2)*cos(pi*rho)
        d2udxy = -pow(pi,2)*x*y*cos(pi*rho)
        g = u
        dg0 = du0
        dg1 = du1
        d2g00 = d2udxx
        d2g11 = d2udyy
        d2g01 = d2udxy
        guu = sqrt(4.0*pow(d2udxy,2)+pow(d2udxx-d2udyy,2))
        guuimp = sqrt(1e-15+4.0*pow(d2udxy,2)+pow(d2udxx-d2udyy,2))
        Theta = pi/4.0+atan(y/x)
        DTheta = 1.0
        truec = 0.0

        # Defining optimal controls for stochastic HJB problem
        def controls(u00,u01,u11):
            gu = sqrt(1e-15+4.0*pow(u01,2)+pow(u00-u11,2))
            gutest = sqrt(4.0*pow(u01,2)+pow(u00-u11,2))
            sinp = (u00-u11)/gu
            cosp = 2.0*u01/gu
            sint = sqrt(3.0)/2.0
            sinpguz = 0.0
            cospguz = 0.0
            sintguz = 0.0
            zeta = signfct(makemax(gutest,0))
            zetau = signfct(makemax(guu,0))
            contsinp = zeta*sinp
            contcosp = zeta*cosp
            contsint = zeta*sint
            c00 = 0.5*(1.0+contsinp*contsint)
            c11 = 0.5*(1.0-contsinp*contsint)
            c01 = 0.5*contcosp*contsint
            cont = [[c00,c01],[c01,c11]]
            fcont = 0.5*(d2udxx+d2udyy)+zetau*(sqrt(3.0)/4.0)*pow(d2udxx-d2udyy,2)/guuimp+zetau*(sqrt(3.0)/2.0)*2.0*pow(d2udxy,2)/guuimp
            return cont, fcont
        MA = False
        truec = 0.0
    elif prob == 1:
        alp = 0.05
        convup = 10.0
        rho = pow(x,2)+pow(y,2)
        u = convup*(0.5*(rho-1.0)-alp*0.25*sin(pi*rho))
        u = u - assemble(u*dx(mesh,degree = quad_deg))/assemble(1.0*dx(mesh,degree = quad_deg))
        du0 = convup*(x - alp*0.5*pi*x*cos(pi*rho))
        du1 = convup*(y - alp*0.5*pi*y*cos(pi*rho))
        d2udxx = convup*(Constant(1.0)-alp*0.5*pi*cos(pi*rho)+alp*pi**2*x**2*sin(pi*rho))
        d2udxy = convup*(Constant(0.0)+alp*pi**2*x*y*sin(pi*rho))
        d2udyy = convup*(Constant(1.0)-alp*0.5*pi*cos(pi*rho)+alp*pi**2*y**2*sin(pi*rho))
        f = 2.0*pow(d2udxx*d2udyy-pow(d2udxy,2.0),0.5)
        g = -u
        dg0 = -du0
        dg1 = -du1
        d2g00 = -d2udxx
        d2g11 = -d2udyy
        d2g01 = -d2udxy
        Theta = 0.0
        DTheta = 0.0
        # Defining optimal controls for the MA-HJB problem
        def controls(u00,u01,u11):
            hh = f
            diff = u00-u11
            c00w = 0.5*(1+(diff)/sqrt(hh**2.0+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c11w = 0.5*(1-(diff)/sqrt(hh**2.0+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c01w = (u01/sqrt(hh**2.0+4.0*pow(u01,2.0)))*sqrt(1-(pow(diff,2.0))/(hh**2+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c00xi = 0.5*(1+diff*sqrt((1-4.0*xi)/(1e-15+4.0*pow(u01,2)+pow(diff,2))))
            c11xi = 0.5*(1-diff*sqrt((1-4.0*xi)/(1e-15+4.0*pow(u01,2)+pow(diff,2))))
            c01xi = signfct(u01)*sqrt(0.25*(1-(pow(diff,2)*(1-4.0*xi))/(1e-15+4*pow(u01,2)+pow(diff,2)))-xi)
            det = 0.25*pow(hh,2)/(pow(hh,2)+pow(diff,2)+4.0*pow(u01,2))
            zeta = signfct(makemax(det-xi,0))
            cont = [[zeta*c00w+(1-zeta)*c00xi,zeta*c01w+(1-zeta)*c01xi],[zeta*c01w+(1-zeta)*c01xi,zeta*c11w+(1-zeta)*c11xi]]
            fcont = -hh*pow(cont[0][0]*cont[1][1]-cont[0][1]*cont[1][0],0.5)
            return cont, fcont
        MA = True
        xi = 1.0/100.0
        truec = 10.0-pi/4



    
    
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
        gradTout = as_vector([u.dx(0)-nouter[0]*(u.dx(0)*nouter[0]+u.dx(1)*nouter[1]),u.dx(1)-nouter[1]*(u.dx(0)*nouter[0]+u.dx(1)*nouter[1])])
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
        deltout = (u.dx(0).dx(0)+u.dx(1).dx(1)-u.dx(0)*nouter[0]-u.dx(1)*nouter[1]\
            -u.dx(0).dx(0)*nouter[0]*nouter[0]\
            -u.dx(1).dx(0)*nouter[1]*nouter[0]\
            -u.dx(0).dx(1)*nouter[0]*nouter[1]\
            -u.dx(1).dx(1)*nouter[1]*nouter[1]\
            )
        return deltout


    def B_star_1(u,v):
        Bstar1 = (u.dx(0).dx(0)*v.dx(0).dx(0)+u.dx(1).dx(0)*v.dx(1).dx(0)+u.dx(0).dx(1)*v.dx(0).dx(1)+u.dx(1).dx(1)*v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
        return Bstar1
    # Curvature and oblique angle dependent bilinear form terms
    def CurvedExtraTerm(u,v):
        extra = (DTheta+1.0)*inner(grad(u),grad(v))*ds(mesh,degree = quad_deg)
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
        Bstar4 =(u.dx(0)*nouter[0]+u.dx(1)*nouter[1])\
            *(delta_Touter(v))\
                *ds(mesh, degree = quad_deg)
        return Bstar4

    def B_star_6(u,v):
        Bstar6 = 0.5*(u.dx(0)*nouter[0]+u.dx(1)*nouter[1])\
            *(v.dx(0)*nouter[0]+v.dx(1)*nouter[1])\
                * ds(mesh, degree = quad_deg)
        return Bstar6
    ### external face sum <grad_T(du/dn),grad_Tv>
    def B_star_7(u,v):
        Bstar7 = inner(grad_T(u.dx(0)*nouter[0]+u.dx(1)*nouter[1]),grad_T(v)) * ds(mesh, degree = quad_deg)
        return Bstar7
    def B_star_5(u,v):
        Bstar5 = DT(inner(beta,grad(u)))*inner(betaperp,grad(v))*ds(mesh, degree = quad_deg)
        return Bstar5

    def B_star(u,v,l,mu):
        Bstar = B_star_1(u,v)+B_star_2_a(u,v)+B_star_2_a(v,u)-B_star_2_b(u,v)-B_star_2_b(v,u)+B_star_3(u,v,mu)+B_star_3(v,u,l)+CurvedExtraTerm(u,v)
        return Bstar
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
        J2exta = muouter*(u.dx(0)-nouter[0]*(u.dx(0)*nouter[0]+u.dx(1)*nouter[1]))*(v.dx(0)-nouter[0]*(v.dx(0)*nouter[0]+v.dx(1)*nouter[1]))*ds(mesh, degree = quad_deg)
        return J2exta
    def J_h_2_ext_b(u,v):
        J2extb = muouter*(u.dx(1)-nouter[1]*(u.dx(0)*nouter[0]+u.dx(1)*nouter[1]))*(v.dx(1)-nouter[1]*(v.dx(0)*nouter[0]+v.dx(1)*nouter[1]))*ds(mesh, degree = quad_deg)
        return J2extb
    def J_h_3_in(u,v):
        J3in = etainner*(u('+')-u('-'))*(v('+')-v('-'))*dS(mesh, degree = quad_deg)
        return J3in
    def J_h_3_ext(u,v,l,mu):
        J3ext = sigma*(inner(beta,grad(u))-l)*(inner(beta,grad(v))-mu)*ds(mesh, degree = quad_deg)
        return J3ext
    def J_h_5_ext(u,v):
        J5ext = sigmajump*((inner(betaperp,grad(u)))*ds(mesh, degree = quad_deg))*((inner(betaperp,grad(v)))*ds(mesh, degree = quad_deg))
        return J5ext
    def J_h_fix(l,mu):
        fix = ellf*(l('+')-l('-'))*(mu('+')-mu('-'))*dS(mesh, degree = quad_deg)
        return fix
    def J_h(u,v,l,mu):
        J = J_h_1(u,v) + J_h_2_in_a(u,v) + J_h_2_in_b(u,v) + J_h_3_in(u,v) + J_h_3_ext(u,v,l,mu) +J_h_3_in(l,mu)+J_h_fix(l,mu)
        return J

    # defining linear problem from one step of Newton's method, requires current iterate and definition of controls.
    def model(init,initc):
        # obtaining nondivergence form operator and right hand side functions from controls definition
        A, rhsf = controls(init.dx(0).dx(0),init.dx(0).dx(1),init.dx(1).dx(1))
 
        # for our examples the coefficient matrix is unit trace, and so it is sufficient to take gamma = 1 as the renormalisation parameter
        gamma = 1.0
        Uold = init
        Cold = initc
        
        # defining nondivergence part of the bilinear form
        def a(u,v):
            a = gamma*(A[0][0]*u.dx(0).dx(0)+A[1][1]*u.dx(1).dx(1)+A[1][0]*u.dx(1).dx(0)+A[0][1]*u.dx(0).dx(1))*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
            return a
        
        # defining bilinear form
        A_gamma = gamma*a(U,v)\
            +theta*B_star(U,v,lamd,mu)\
            +(1-theta)*Deltainner(U,v)\
            +J_h(U,v,lamd,mu)\
            -Deltainner(U,v)\
        
        # defining linear form
        L = gamma*rhsf*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh, degree = quad_deg)
        Uh = Function(S)

        # implementing nullspace, as solution should have zero sum
        V_basis = VectorSpaceBasis(constant=True)
        nullspace = MixedVectorSpaceBasis(S, [V_basis, S[1]])

        # begin timing of linear system solve
        ttt = time()
        # solving linear system
        solve(A_gamma == L,Uh,nullspace = nullspace,
          solver_parameters = {"mat_type": "aij",
          "snes_type": "newtonls",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "snes_monitor": True,
          "snes_rtol": 1e-16,
          "snes_atol": 1e-25})
        # end timing of linear system solve

        tt = (time()-ttt)

        ndofs = Uh.vector().array().size
        uh, ch = Uh

        # calculating Newton step error
        diff = abs(uh-Uold)
        L2relerr = sqrt(assemble(pow(uh-Uold,2)*dx(mesh,degree = quad_deg)))
        Crelerr  = sqrt(assemble(pow(ch-Cold,2)*ds(mesh,degree = quad_deg)))
        return uh, ch, L2relerr, resid, ndofs, diff, tt, Crelerr

    L2relerr = 1
    C2relerr = 1
    resid = 1
    i = 0
    initialdist = [];
    newtoncount = [];
    newtonerr = [];
    newtontimes = [];
    U0, C0 = Constant(0.0), project(Constant(1.0),Con1)
    Unew, Cnew = project(Constant(0.0),FES), project(Constant(0.0),Con1)
    
    # Defining Newton iteration: implemented to stop if Newton step error falls below tolerance, or if iteration max is exceeded
    # Begin timing Newton's method
    ttt = time()
    while i < newtonitermax and L2relerr > newtontol:
        Unew, Cnew, L2relerr, resid, ndofs, diff, t, C2relerr = model(Unew,Cnew)
        print("C increment error = ",C2relerr)
        print("u increment error = ",L2relerr)
        newtoncount.append(i)
        newtonerr.append(L2relerr)
        newtontimes.append(t)
        i += 1
    # End timing Newton's method
    tt = time()-ttt

    U, ch = Unew, Cnew
    Uuavg = assemble((U-u)*dx(mesh, degree = quad_deg))/assemble(1.0*dx(mesh, degree = quad_deg))
    MAUuavg = assemble((U+u)*dx(mesh, degree = quad_deg))/assemble(1.0*dx(mesh, degree = quad_deg))


    
    # Calculating difference between initial guess and final Newton iterate: \|u_h^0-u_h^N\|_{h,1}
    initialdist.append((sqrt(assemble((pow((U-U0).dx(0).dx(0),2.0)+pow((U-U0).dx(0).dx(1),2.0)+pow((U-U0).dx(1).dx(0),2.0)+pow((U-U0).dx(1).dx(1),2.0))*dx(mesh, degree = quad_deg)+J_h((U-U0),(U-U0),ch-C0,ch-C0)+0.5*(pow(((U-U0).dx(0))*n[0]+((U-U0).dx(1))*n[1],2.0)*ds(mesh,degree = quad_deg))))))

    # record NDoFs
    ndof = ndofs

    #calculating errors
    if MA == True:
        ch = ch + truec
        e_L2 = (sqrt(assemble((pow(U+u-MAUuavg,2.0))*dx(mesh, degree = quad_deg))))
        e_H1 = (sqrt(assemble((pow(U.dx(0)+du0,2.0)+pow(U.dx(1)+du1,2.0))*dx(mesh, degree = quad_deg))))
        e_H2 = (sqrt(assemble(((pow(U.dx(0).dx(0)+d2udxx,2.0)+pow(U.dx(0).dx(1)+d2udxy,2.0)+pow(U.dx(1).dx(0)+d2udxy,2.0)+pow(U.dx(1).dx(1)+d2udyy,2.0))*dx(mesh, degree = quad_deg)))))
        e_h1  = (sqrt(assemble((pow(U.dx(0).dx(0)+d2udxx,2.0)+pow(U.dx(0).dx(1)+d2udxy,2.0)+pow(U.dx(1).dx(0)+d2udxy,2.0)+pow(U.dx(1).dx(1)+d2udyy,2.0))*dx(mesh, degree = quad_deg)+cstab*J_h(U+u,U+u,ch,ch)+cstab*(pow((U+u).dx(0),2)+pow((U+u).dx(1),2))*ds(mesh, degree = quad_deg))))
        e_C = (assemble(abs(ch)*ds(mesh,degree = quad_deg))/assemble(1.0*ds(mesh,degree = quad_deg)))
    else:
        e_L2 = (sqrt(assemble((pow(U-u-Uuavg,2.0))*dx(mesh, degree = quad_deg))))
        e_H1 = (sqrt(assemble((pow(U.dx(0)-du0,2.0)+pow(U.dx(1)-du1,2.0))*dx(mesh, degree = quad_deg))))
        e_H2 = (sqrt(assemble(((pow(U.dx(0).dx(0)-d2udxx,2.0)+pow(U.dx(0).dx(1)-d2udxy,2.0)+pow(U.dx(1).dx(0)-d2udxy,2.0)+pow(U.dx(1).dx(1)-d2udyy,2.0))*dx(mesh, degree = quad_deg)))))
        e_h1 = (sqrt(assemble((pow(U.dx(0).dx(0)-d2udxx,2.0)+pow(U.dx(0).dx(1)-d2udxy,2.0)+pow(U.dx(1).dx(0)-d2udxy,2.0)+pow(U.dx(1).dx(1)-d2udyy,2.0))*dx(mesh, degree = quad_deg)+cstab*J_h(U-u,U-u,ch,ch)+cstab*(pow((U-u).dx(0),2)+pow((U-u).dx(1),2))*ds(mesh, degree = quad_deg))))
        e_C = (assemble(abs(ch-truec)*ds(mesh,degree = quad_deg))/assemble(1.0*ds(mesh,degree = quad_deg)))
    # record mesh size
    h = sqrt(max(interpolate(CellVolume(mesh), DG).vector()[:]))
    hm.append(h)

    # returning Newton iterations and step errors
    return newtoncount, newtonerr, newtontimes, e_L2, e_H1, e_H2, e_h1, e_C, tt, hm, ndofs, initialdist

for deg in [4]:
    e_L2 = []; e_H1 = []; e_H2 = []; e_h1 = []; e_C = []; EOCL2 = []; EOCH1 = []; EOCH2 = []; EOCh1 = []; EOCC = []; newtoncounts = []; newtonerrs = []; hm = []; tt = []; ndof = []; dists = []; ntotlist = [];
    EOCL2.append(0); EOCH1.append(0); EOCH2.append(0); EOCh1.append(0); EOCC.append(0);
    for idx in [6]:
        newtoncount, newtonerr, newtontimes, e_L21, e_H11, e_H21, e_h11, e_C1, tt1, hm1, ndofs1, indist = Newton(idx,pow(deg,2),0.15625*6.0*pow(deg,4)/8.0,1.0,1.0,50,2.5)
        ntot = [];
        e_L2.append(e_L21); e_H1.append(e_H11); e_H2.append(e_H21); e_h1.append(e_H21); e_C.append(e_C1); ndof.append(ndofs1); tt.append(tt1); dists.append(indist); ntot.append(newtoncount[len(newtoncount)-1]); ntotlist.append(newtoncount[len(newtoncount)-1]);
        # Saving Newton iteration data to file
        out_name11 = "MA-HJB-oblique-data/Newtonstepsrefinement_" + str(idx) + "p_" + str(deg) + ".txt"
        out_name12 = "MA-HJB-oblique-data/Newtonincrementerrorrefinement_" + str(idx) + "p_" +str(deg)+ ".txt"
        out_name13 = "MA-HJB-oblique-data/Newtontimesrefinement_" + str(idx) + "p_" +str(deg)+ ".txt"
        out_name16 = "MA-HJB-oblique-data/Newtoninitdistrefinement_" + str(idx) + "p_" +str(deg)+ ".txt"
        out_name17 = "MA-HJB-oblique-data/NewtonTotalstepsrefinement_" + str(idx) + "p_" + str(deg) + ".txt"
        np.savetxt(out_name11,newtoncount,fmt = '%s')
        np.savetxt(out_name12,newtonerr,fmt = '%s')
        np.savetxt(out_name13,newtontimes,fmt = '%s')
        np.savetxt(out_name16,indist,fmt = '%s')
        np.savetxt(out_name17,ntot,fmt = '%s')
        print("Newton steps:",newtoncount[len(newtoncount)-1])
        print("Increment errors:",newtonerr)

    print("polynomial degree: ", deg)
    #Calcuating error orders of convergence.
    for k in range(1,len(e_L2)):
        EOCL2.append(ln(e_L2[k-1]/(e_L2[k]))/ln(hm[k-1]/hm[k]))
        EOCH1.append(ln(e_H1[k-1]/(e_H1[k]))/ln(hm[k-1]/hm[k]))
        EOCH2.append(ln(e_H2[k-1]/(e_H2[k]))/ln(hm[k-1]/hm[k]))
        EOCh1.append(ln(e_h1[k-1]/(e_h1[k]))/ln(hm[k-1]/hm[k]))
        EOCC.append(ln(e_C[k-1]/(e_C[k]))/ln(hm[k-1]/hm[k]))
    k = 0
    for k in range(len(e_L2)):
        print("Number of DOFs = ", ndof[k])
        print("||u - u_h||_0 = ", e_L2[k], "   EOC = ", EOCL2[k])
        print("||u - u_h||_1 = ", e_H1[k], "   EOC = ", EOCH1[k])
        print("||u - u_h||_2 = ", e_H2[k], "   EOC = ", EOCH2[k])
        print("||u - u_h||_h,1 = ", e_h1[k], "   EOC = ", EOCh1[k])
        print("||c - c_h||_0 = ", e_C[k], "  EOC = ",EOCC[k])
        k = k+1

    # saving data to file
    out_name1 = "MA-HJB-oblique-data/e_L2p%i.txt" %deg
    out_name2 = "MA-HJB-oblique-data/e_H1p%i.txt" %deg
    out_name3 = "MA-HJB-oblique-data/e_H2p%i.txt" %deg
    out_name4 = "MA-HJB-oblique-data/e_h-1p%i.txt" %deg
    out_name5 = "MA-HJB-oblique-data/EOCL2p%i.txt" %deg
    out_name6 = "MA-HJB-oblique-data/EOCH1p%i.txt" %deg
    out_name7 = "MA-HJB-oblique-data/EOCH2p%i.txt" %deg
    out_name8 = "MA-HJB-oblique-data/EOCh-1p%i.txt" %deg
    out_name9 = "MA-HJB-oblique-data/dofsp%i.txt" %deg
    out_name10 = "MA-HJB-oblique-data/timesp%i.txt" %deg
    out_name11 = "MA-HJB-oblique-data/e_Cp%i.txt" %deg
    out_name12 = "MA-HJB-oblique-data/EOCCp%i.txt" %deg
    np.savetxt(out_name1,e_L2,fmt = '%s')
    np.savetxt(out_name2,e_H1,fmt = '%s')
    np.savetxt(out_name3,e_H2,fmt = '%s')
    np.savetxt(out_name4,e_h1,fmt = '%s')
    np.savetxt(out_name5,EOCL2,fmt = '%s')
    np.savetxt(out_name6,EOCH1,fmt = '%s')
    np.savetxt(out_name7,EOCH2,fmt = '%s')
    np.savetxt(out_name8,EOCh1,fmt = '%s')
    np.savetxt(out_name9,ndof,fmt = '%s')
    np.savetxt(out_name10,tt,fmt = '%s')
    np.savetxt(out_name11,e_C,fmt = '%s')
    np.savetxt(out_name12,EOCC,fmt = '%s')

out_name13 = "MA-HJB-oblique-data/meshsize.txt"
np.savetxt(out_name13,hm,fmt = '%s')
print("experiment no. : ", prob)
