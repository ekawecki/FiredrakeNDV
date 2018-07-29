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
quad_deg = 20
prob = 6
refinementsno = 7
newtonitermax = 30
newtontol = 1e-12
cstab = 2.5
disk = True
inhomogeneous = False

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

def Newton(idx,mucon,etacon,quad_deg,cstab):
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

    # finite element space
    FES = FunctionSpace(mesh,"DG",deg)
    theta = 0.5

    # defining maximum mesh size
    DG = FunctionSpace(mesh, "DG", 0);
    h = sqrt(max(interpolate(CellVolume(mesh), DG).vector()[:]))

    # defining local mesh size
    hfct = sqrt(CellVolume(mesh))

    # coordinate functions
    x, y = SpatialCoordinate(mesh)
    
    # defining jump stabilisation parameters
    muinner = cstab*mucon*pow(makemin(hfct('+'),hfct('-')),-1)
    muouter = cstab*mucon*pow(hfct,-1)
    etainner = cstab*etacon*pow(makemin(hfct('+'),hfct('-')),-3)
    etaouter = cstab*etacon*pow(hfct,-3)
    
    # defining benchmark solutions and nonlinear problems
    if prob == 0:
        rho = pow(x,2)+pow(y,2)
        u = (1.0/50.0)*(50.0*(rho-1.0)+0.3*pow(rho-1.0,3.0))
        ubc = -u
        du0 = (1.0/50.0)*(100*x+1.8*x*pow(rho-1,2.0))
        du0bc = -du0
        du1 = (1.0/50.0)*(100*y+1.8*y*pow(rho-1,2.0))
        du1bc = -du1
        d2udxx = (1.0/50.0)*(Constant(100.0)+1.8*pow(rho-1,2.0)+7.2*pow(x,2.0)*(rho-1))
        d2udxy = (1.0/50.0)*(7.2*x*y*(rho-1))
        d2udyy = (1.0/50.0)*(Constant(100.0)+1.8*pow(rho-1,2.0)+7.2*pow(y,2.0)*(rho-1))
        gamma = 1.0
        f = 2.0*pow(d2udxx*d2udyy-pow(d2udxy,2.0),0.5)
        g = u
        dg0 = du0
        dg1 = du1
        d2g00 = d2udxx
        d2g11 = d2udyy
        d2g01 = d2udxy
        MA = True
        xi = 1.0/100.0
        # Defining optimal controls for the MA-HJB problem
        def controls(u00,u01,u11):
            hh = f
            diff = u00-u11
            c00w = 0.5*(1+diff/sqrt(hh**2.0+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c11w = 0.5*(1-diff/sqrt(hh**2.0+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c01w = (u01/sqrt(hh**2.0+4.0*pow(u01,2.0)))*sqrt(1-(pow(diff,2.0))/(hh**2+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c00xi = 0.5*(1+diff*sqrt((1-4.0*xi)/(1e-15+4.0*pow(u01,2)+pow(diff,2))))
            c11xi = 0.5*(1-diff*sqrt((1-4.0*xi)/(1e-15+4.0*pow(u01,2)+pow(diff,2))))
            c01xi = signfct(u01)*sqrt(0.25*(1-(pow(diff,2)*(1-4.0*xi))/(1e-15+4*pow(u01,2)+pow(diff,2)))-xi)
            det = 0.25*pow(hh,2)/(pow(hh,2)+pow(diff,2)+4.0*pow(u01,2))
            zeta = signfct(makemax(det-xi,0))
            cont = [[zeta*c00w+(1-zeta)*c00xi,zeta*c01w+(1-zeta)*c01xi],[zeta*c01w+(1-zeta)*c01xi,zeta*c11w+(1-zeta)*c11xi]]
            fcont = -hh*pow(cont[0][0]*cont[1][1]-cont[0][1]*cont[1][0],0.5)
            return cont, fcont
    if prob == 1:
        rho = 2.0-pow(x,2)-pow(y,2)
        u = -pow(rho,0.5)+0.5*(pow(x,2)+pow(y,2)-1)
        du0 = x/sqrt(rho)+x
        du1 = y/sqrt(rho)+y
        d2udxx = (2.0-pow(y,2.0))/pow(rho,3.0/2.0)+1.0
        d2udxy = x*y/pow(rho,3.0/2.0)
        d2udyy = (2.0-pow(x,2.0))/pow(rho,3.0/2.0)+1.0
        gamma = 1.0
        f = 2.0*pow(d2udxx*d2udyy-pow(d2udxy,2.0),0.5)
        g = -u
        dg0 = -du0
        dg1 = -du1
        d2g00 = -d2udxx
        d2g11 = -d2udyy
        d2g01 = -d2udxy
        # Defining optimal controls for the MA-HJB problem
        def controls(u00,u01,u11):
            hh = f
            diff = u00-u11
            c00 = 0.5*(1+abs(diff)/sqrt(hh**2.0+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c11 = 0.5*(1-abs(diff)/sqrt(hh**2.0+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c01 = (u01/sqrt(hh**2.0+4.0*pow(u01,2.0)))*sqrt(1-(pow(diff,2.0))/(hh**2+pow(diff,2.0)+4.0*pow(u01,2.0)))
            cont = [[c00,c01],[c01,c11]]
            return cont
        MA = True
        xi = 1.0/100.0
    elif prob == 2:
        alp = 0.05
        convup = 10.0
        rho = pow(x,2)+pow(y,2)
        u = convup*(0.5*(rho-1.0)-alp*0.25*sin(pi*rho))
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
            fcont = -hh*pow(cont[0][0]*cont[1][1]-cont[1][0]*cont[0][1],0.5)
            return cont, fcont
        MA = True
        xi = 1.0/100.0
    elif prob == 3:
        rho = pow(x,2)+pow(y,2)
        u = 0.1*(pow(rho+1.0,2)-4.0)
        du0 = 0.1*2.0*(2.0*x)*rho
        du1 = 0.1*2.0*(2.0*y)*rho
        d2udxx = 0.1*(4.0*rho+2.0*(2.0*x)*2.0*x)
        d2udxy = 0.1*2.0*(2.0*x)*2.0*y
        d2udyy = 0.1*(4.0*rho+2.0*(2.0*y)*2.0*y)
        f = 2.0*pow(d2udxx*d2udyy-pow(d2udxy,2.0),0.5)
        g = -u
        dg0 = -du0
        dg1 = -du1
        d2g00 = -d2udxx
        d2g11 = -d2udyy
        d2g01 = -d2udxy
        # Defining optimal controls for the MA-HJB problem
        def controls(u00,u01,u11):
            hh = f
            diff = u00-u11
            c00 = 0.5*(1+abs(diff)/sqrt(hh**2.0+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c11 = 0.5*(1-abs(diff)/sqrt(hh**2.0+pow(diff,2.0)+4.0*pow(u01,2.0)))
            c01 = (u01/sqrt(hh**2.0+4.0*pow(u01,2.0)))*sqrt(1-(pow(diff,2.0))/(hh**2+pow(diff,2.0)+4.0*pow(u01,2.0)))
            cont = [[c00,c01],[c01,c11]]
            return cont
        MA = True
        xi = 1.0/100.0
    elif prob == 4:
        # Linear HJB problem
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
        d2g11 = d2udyy
        d2g01 = d2udxy
    elif prob == 5:
        # Linear HJB problem
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
        g = -u
        dg0 = -du0
        dg1 = -du1
        d2g00 = -d2udxx
        d2g11 = -d2udyy
        d2g01 = -d2udxy
    elif prob == 6:
        rho = pow(x,2)+pow(y,2)
        alp = 1.0
        u = alp*0.25*sin(pi*rho)
        du0 = alp*0.5*pi*x*cos(pi*rho)
        du1 = alp*0.5*pi*y*cos(pi*rho)
        d2udxx = alp*(0.5*pi*cos(pi*rho)-pi**2*x**2*sin(pi*rho))
        d2udxy = alp*(-pi**2*x*y*sin(pi*rho))
        d2udyy = alp*(0.5*pi*cos(pi*rho)-pi**2*y**2*sin(pi*rho))
        g = u
        dg0 = du0
        dg1 = du1
        d2g00 = d2udxx
        d2g11 = d2udyy
        d2g01 = d2udxy
        guu = sqrt(4.0*pow(d2udxy,2)+pow(d2udxx-d2udyy,2))
        guuimp = sqrt(1e-15+4.0*pow(d2udxy,2)+pow(d2udxx-d2udyy,2))
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
    elif prob == 7:
        u = sin(pi*x)*sin(pi*y)
        du0 = pi*cos(pi*x)*sin(pi*y)
        du1 = pi*sin(pi*x)*cos(pi*y)
        d2udxx = -pi*pi*sin(pi*x)*sin(pi*y)
        d2udxy = pi*pi*cos(pi*x)*cos(pi*y)
        d2udyy = -pi*pi*sin(pi*x)*sin(pi*y)
        guu = sqrt(4.0*pow(d2udxy,2)+pow(d2udxx-d2udyy,2))
        guuimp = sqrt(1e-15+4.0*pow(d2udxy,2)+pow(d2udxx-d2udyy,2))
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
    # tangential gradient function
    def grad_T(u):
        gradT = as_vector([u.dx(0)-n[0]*(u.dx(0)*n[0]+u.dx(1)*n[1]),u.dx(1)-n[1]*(u.dx(0)*n[0]+u.dx(1)*n[1])])
        return gradT

    # beginning definition of bilinear form
    def B_star_1(u,v):
        Bstar1 = (u.dx(0).dx(0)*v.dx(0).dx(0)+u.dx(1).dx(0)*v.dx(1).dx(0)+u.dx(0).dx(1)*v.dx(0).dx(1)+u.dx(1).dx(1)*v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
        return Bstar1

    # Curvature dependent bilinear form terms
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
                    *dS(mesh, degree = quad_deg)
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
                    *dS(mesh, degree = quad_deg)
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
                *ds(mesh, degree = quad_deg)
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
            * ds(mesh, degree = quad_deg)
        return Bstar4b
               
    # defining bilinear form B_{h,*}
    def B_star(u,v):
        Bstar = B_star_1(u,v)+B_star_2_a(u,v)+B_star_2_b(u,v)-B_star_3_a(u,v)-B_star_3_b(u,v)-B_star_4_a(u,v)-B_star_4_b(u,v)+CurvedExtraTerm(u,v)
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
    def J_h_3_ext(u,v):
        J3ext = etaouter*u*v*ds(mesh, degree = quad_deg)
        return J3ext

    def J_h(u,v):
        J = J_h_1(u,v) + J_h_2_in_a(u,v) + J_h_2_in_b(u,v) + J_h_2_ext_a(u,v) + J_h_2_ext_b(u,v) + J_h_3_in(u,v) + J_h_3_ext(u,v)
        return J





    # defining linear problem from one step of Newton's method, requires current iterate and definition of controls.
    def model(init):
        # obtaining nondivergence form operator and right hand side functions from controls definition
        A, rhsf = controls(init.dx(0).dx(0),init.dx(0).dx(1),init.dx(1).dx(1))
        
        # for our examples the coefficient matrix is unit trace, and so it is sufficient to take gamma = 1 as the renormalisation parameter
        gamma = 1.0
        
        # defining previous Newton step for calculation of increment error
        Uold = init
        
        # defining nondivergence part of the bilinear form
        def a(u,v):
            a = gamma*(A[0][0]*u.dx(0).dx(0)+A[1][1]*u.dx(1).dx(1)+A[1][0]*u.dx(1).dx(0)+A[0][1]*u.dx(0).dx(1))*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
            return a
        
        U = TrialFunction(FES)
        v = TestFunction(FES)
        
        # defining bilinear form
        A_gamma = a(U,v)\
            +theta*B_star(U,v) \
            +(1-theta)*Deltainner(U,v)\
            +J_h(U,v)\
            -Deltainner(U,v)
        
        # defining linear form
        L = gamma*rhsf*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh, degree = quad_deg)

        # altering the linear form for the inhomogeneous Dirichlet case
        if inhomogeneous == True:
            L= gamma*(f)*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh,degree = quad_deg)\
                +etaouter*g*v*ds(mesh,degree = quad_deg)\
                    +(muouter+0.5*bdry_indicator)*inner(as_vector([dg0-n[0]*(dg0*n[0]+dg1*n[1]),dg1-n[1]*(dg0*n[0]+dg1*n[1])]),grad_T(v))*ds(mesh,degree = quad_deg)\
                        - 0.5*(inner(grad_T(v.dx(0)*n[0]+v.dx(1)*n[1]),grad_T(g)))*ds(mesh,degree = quad_deg)\
                            - 0.5*((d2g00+d2g11-bdry_indicator*(dg0*n[0]+dg1*n[1])-(d2g00*n[0]*n[0]+2.0*d2g01*n[0]*n[1]+d2g11*n[1]*n[1])))*(v.dx(0)*n[0]+v.dx(1)*n[1])*ds(mesh,degree = quad_deg)
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
        tt = (time()-t)

        # getting number of DoFs
        ndofs = U.vector().array().size

        # calculating Newton step error
        diff = abs(U-Uold)
        L2relerr = sqrt(assemble(pow(U-Uold,2)*dx(mesh,degree = quad_deg)))

        return U, L2relerr, resid, ndofs, diff, tt
            
    # providing initial guess for Newton's method
    U = project(cos(pi*x)*cos(pi*y)-1,FES)
    U0 = U

    L2relerr = 1
    resid = 1
    i = 1
    ndofs = 0
    t = 0
    initialdist = [];
    newtoncount = [];
    newtonerr = [];
    newtontimes = [];
    # Defining Newton iteration: implemented to stop if Newton step error falls below tolerance, or if iteration max is exceeded
    # Begin timing Newton's method
    ttt = time()
    while i < newtonitermax and L2relerr > newtontol:
        U, L2relerr, resid, ndofs, diff, t = model(U)
        newtoncount.append(i)
        newtonerr.append(L2relerr)
        newtontimes.append(t)
        i += 1
    # End timing Newton's method
    tt = time()-ttt

    # Calculating difference between initial guess and final Newton iterate: \|u_h^0-u_h^N\|_{h,1}
    initialdist.append((sqrt(assemble((pow((U-U0).dx(0).dx(0),2.0)+pow((U-U0).dx(0).dx(1),2.0)+pow((U-U0).dx(1).dx(0),2.0)+pow((U-U0).dx(1).dx(1),2.0))*dx(mesh, degree = quad_deg)+J_h((U-U0),(U-U0))+0.5*(pow(((U-U0).dx(0))*n[0]+((U-U0).dx(1))*n[1],2.0)*ds(mesh,degree = quad_deg))))))

    # record NDoFs
    ndof = ndofs

    # record errors
    if MA == True:
        e_L2 = (sqrt(assemble((pow(U+u,2.0))*dx(mesh, degree = quad_deg))))
        e_H1 = (sqrt(assemble((pow(U.dx(0)+du0,2.0)+pow(U.dx(1)+du1,2.0))*dx(mesh, degree = quad_deg))))
        e_H2 = (sqrt(assemble(((pow(U.dx(0).dx(0)+d2udxx,2.0)+pow(U.dx(0).dx(1)+d2udxy,2.0)+pow(U.dx(1).dx(0)+d2udxy,2.0)+pow(U.dx(1).dx(1)+d2udyy,2.0))*dx(mesh, degree = quad_deg)))))
        e_h1 = (sqrt(assemble((pow(U.dx(0).dx(0)+d2udxx,2.0)+pow(U.dx(0).dx(1)+d2udxy,2.0)+pow(U.dx(1).dx(0)+d2udxy,2.0)+pow(U.dx(1).dx(1)+d2udyy,2.0))*dx(mesh, degree = quad_deg)+J_h(U+u,U+u)+0.5*(pow((U.dx(0)+du0)*n[0]+(U.dx(1)+du1)*n[1],2.0)*ds(mesh,degree = quad_deg)))))
    else:
        e_L2 = (sqrt(assemble((pow(U-u,2.0))*dx(mesh, degree = quad_deg))))
        e_H1 = (sqrt(assemble((pow(U.dx(0)-du0,2.0)+pow(U.dx(1)-du1,2.0))*dx(mesh, degree = quad_deg))))
        e_H2 = (sqrt(assemble(((pow(U.dx(0).dx(0)-d2udxx,2.0)+pow(U.dx(0).dx(1)-d2udxy,2.0)+pow(U.dx(1).dx(0)-d2udxy,2.0)+pow(U.dx(1).dx(1)-d2udyy,2.0))*dx(mesh, degree = quad_deg)))))
        if disk == True:
            e_h1 = (sqrt(assemble((pow(U.dx(0).dx(0)-d2udxx,2.0)+pow(U.dx(0).dx(1)-d2udxy,2.0)+pow(U.dx(1).dx(0)-d2udxy,2.0)+pow(U.dx(1).dx(1)-d2udyy,2.0))*dx(mesh, degree = quad_deg)+J_h(U-u,U-u)+0.5*(pow((U.dx(0)-du0)*n[0]+(U.dx(1)-du1)*n[1],2.0)*ds(mesh,degree = quad_deg)))))
        else:
            e_h1 = (sqrt(assemble((pow(U.dx(0).dx(0)-d2udxx,2.0)+pow(U.dx(0).dx(1)-d2udxy,2.0)+pow(U.dx(1).dx(0)-d2udxy,2.0)+pow(U.dx(1).dx(1)-d2udyy,2.0))*dx(mesh, degree = quad_deg)+J_h(U-u,U-u))))
    # record mesh size
    h = sqrt(max(interpolate(CellVolume(mesh), DG).vector()[:]))
    hm.append(h)

    # returning Newton iterations and step errors
    return newtoncount, newtonerr, newtontimes, e_L2, e_H1, e_H2, e_h1, tt, hm, ndofs, initialdist
for deg in [2,3,4]:
    e_L2 = []; e_H1 = []; e_H2 = []; e_h1 = []; EOCL2 = []; EOCH1 = []; EOCH2 = []; EOCh1 = []; newtoncounts = []; newtonerrs = []; hm = []; tt = []; ndof = []; dists = []; ntotlist = [];
    EOCL2.append(0); EOCH1.append(0); EOCH2.append(0); EOCh1.append(0);
    for idx in range(refinementsno):
        newtoncount, newtonerr, newtontimes,e_L21, e_H11, e_H21, e_h11, tt1, hm1, ndofs1, indist = Newton(idx,pow(deg-1,2)/2.0,3.0*pow(deg-1,4)/8.0,8,2.0)
        print(indist)
        ntot = [];
        e_L2.append(e_L21); e_H1.append(e_H11); e_H2.append(e_H21); e_h1.append(e_H21); ndof.append(ndofs1); tt.append(tt1); dists.append(indist); ntot.append(newtoncount[len(newtoncount)-1]); ntotlist.append(newtoncount[len(newtoncount)-1]);
        # Saving Newton iteration data to file
        out_name11 = "MA-HJB-Dirichlet-data/Newtonstepsrefinement_" + str(idx) + "p_" + str(deg) + ".txt"
        out_name12 = "MA-HJB-Dirichlet-data/Newtonincrementerrorrefinement_" + str(idx) + "p_" +str(deg)+ ".txt"
        out_name13 = "MA-HJB-Dirichlet-data/Newtontimesrefinement_" + str(idx) + "p_" +str(deg)+ ".txt"
        out_name16 = "MA-HJB-Dirichlet-data/Newtoninitdistrefinement_" + str(idx) + "p_" +str(deg)+ ".txt"
        out_name17 = "MA-HJB-Dirichlet-data/NewtonTotalstepsrefinement_" + str(idx) + "p_" + str(deg) + ".txt"
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
    k = 0
    for k in range(len(e_L2)):
        print("Number of DOFs = ", ndof[k])
        print("||u - u_h||_0 = ", e_L2[k], "   EOC = ", EOCL2[k])
        print("||u - u_h||_1 = ", e_H1[k], "   EOC = ", EOCH1[k])
        print("||u - u_h||_2 = ", e_H2[k], "   EOC = ", EOCH2[k])
        print("||u - u_h||_h,1 = ", e_h1[k], "   EOC = ", EOCh1[k])
        k = k+1

    # saving data to file
    out_name1 = "MA-HJB-Dirichlet-data/e_L2p%i.txt" %deg
    out_name2 = "MA-HJB-Dirichlet-data/e_H1p%i.txt" %deg
    out_name3 = "MA-HJB-Dirichlet-data/e_H2p%i.txt" %deg
    out_name4 = "MA-HJB-Dirichlet-data/e_h-1p%i.txt" %deg
    out_name5 = "MA-HJB-Dirichlet-data/EOCL2p%i.txt" %deg
    out_name6 = "MA-HJB-Dirichlet-data/EOCH1p%i.txt" %deg
    out_name7 = "MA-HJB-Dirichlet-data/EOCH2p%i.txt" %deg
    out_name8 = "MA-HJB-Dirichlet-data/EOCh-1p%i.txt" %deg
    out_name9 = "MA-HJB-Dirichlet-data/dofsp%i.txt" %deg
    out_name10 = "MA-HJB-Dirichlet-data/timesp%i.txt" %deg
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

out_name11 = "MA-HJB-Dirichlet-data/meshsize.txt"
np.savetxt(out_name11,hm,fmt = '%s')
print("experiment no. : ", prob)


