import numpy as np
import sympy as sym
from sympy.abc import t
%matplotlib inline
import matplotlib.pyplot as plt
import scipy.linalg
import time

#######################
# Custom latex printing
def custom_latex_printer(exp,**options):
    from google.colab.output._publish import javascript
    url = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/latest.js?config=TeX-AMS_HTML"
    javascript(url=url)
    return sym.printing.latex(exp,**options)
sym.init_printing(use_latex="mathjax",latex_printer=custom_latex_printer)

####################
# Simulation helpers
def integrate(f,x0,dt): # I changed this to include explicit time-varying dynamics
    """
    This function takes in an initial condition x0 and a timestep dt,
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. It outputs a vector x at the future time step.
    """
    
    k1=dt*f(x0) # we call the dynamics by passing t directly now
    k2=dt*f(x0+k1/2.)
    k3=dt*f(x0+k2/2.)
    k4=dt*f(x0+k3)
    xnew=x0+(1/6.)*(k1+2.*k2+2.*k3+k4)
    return xnew 

############
# Simulating
def simulate(f,x0,tspan,dt, phi_condition=lambda x: False, impact_update=lambda x: x):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. Additionally, it takes in a function
    phi_condition(x) that evaluates whether or not a system is undergoing
    impacts. Finally, impact_update(x) is a function that applies the 
    update law. simulate_impact outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).
    """
    N = int((max(tspan)-min(tspan))/dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan),max(tspan),N)
    xtraj = np.zeros((len(x0),N))
    for i in range(N):
        if phi_condition(x):
            xtraj[:,i]=integrate(f,impact_update(x),dt) # I pass the current time to the integrator now
        else:
            xtraj[:,i]=integrate(f,x,dt)
        x = np.copy(xtraj[:,i])
    return xtraj

def simulate_impacts1(f,x0,dt,phi_cond,impact_update):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. Additionally, this includes a flag (default false)
    that allows one to supply an Euler intergation scheme instead of 
    the given scheme. It outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).
    """
    N = 1000
    x = np.copy(x0)
    xtraj = np.zeros((len(x0),N))
    count=0
    time1=0
    for i in range(N):
        if count< 1:
          ce=phi_cond(x)
          if ce>0 :
            xtraj[:,i]=integrate(f,impact_update(x,ce),dt)
            x = np.copy(xtraj[:,i])
            count=count+1
            time1=time1+dt
          else :
            xtraj[:,i]=integrate(f,x,dt)
            x = np.copy(xtraj[:,i])
            time1=time1+dt
    A=int((time1)/dt)
    tvec = np.linspace(0,time1,A)
    xtraj=np.copy(xtraj[:,0:A])
    return xtraj, A ,time1

def simulate_impacts(f,x0,tspan,dt,phi_cond,impact_update):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. Additionally, this includes a flag (default false)
    that allows one to supply an Euler intergation scheme instead of 
    the given scheme. It outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).
    """
    N = int((max(tspan)-min(tspan))/dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan),max(tspan),N)
    xtraj = np.zeros((len(x0),N))
    for i in range(N):
      ce=phi_cond(x)
      if ce>0 :
        xtraj[:,i]=integrate(f,impact_update(x,ce),dt)
        x = np.copy(xtraj[:,i])
      else :
        xtraj[:,i]=integrate(f,x,dt)
        x = np.copy(xtraj[:,i])
    return xtraj 

##################
# Helper Functions
def rectangle(xc,yc,tht,ll,ww):
    wax = np.array([0,0,1])
    rotMat = T(wax,tht,np.array([xc,yc,0]), False)
    p1 = np.dot(rotMat,T(wax,0,np.array([ll/2.,ww/2.,0]), False))
    p2 = np.dot(rotMat,T(wax,0,np.array([-ll/2.,ww/2.,0]), False))
    p3 = np.dot(rotMat,T(wax,0,np.array([-ll/2.,-ww/2.,0]), False))
    p4 = np.dot(rotMat,T(wax,0,np.array([ll/2.,-ww/2.,0]), False))
    return np.squeeze(np.array([p1[0,3],p1[1,3],p2[0,3],p2[1,3],
                                p3[0,3],p3[1,3],p4[0,3],p4[1,3]],dtype=np.float64))

def hat(w,use_sym=True):
    if use_sym:
        what = sym.Matrix([[    0,-w[2], w[1]],
                           [ w[2],    0,-w[0]],
                           [-w[1], w[0],   0]])
    else:
        what = np.array([[    0,-w[2], w[1]],
                         [ w[2],    0,-w[0]],
                         [-w[1], w[0],   0]])
    return what

def unhat(what,use_sym=True):
    if use_sym:
        w = sym.Matrix([what[2,1],what[0,2],what[1,0]])
    else:
        w = np.array([what[2,1],what[0,2],what[1,0]])
    return w

def rot(w,theta,use_sym=True):
    if use_sym:
        rotMat = sym.Matrix(sym.simplify(sym.exp(hat(w,use_sym)*theta)))
        for i in range(rotMat.shape[0]):
            for j in range(rotMat.shape[1]):
                rotMat[i,j] = sym.simplify(rotMat[i,j].rewrite(sym.sin)) # Simplification made because sympy seems to have
                                                                         # issues applying the Euler identity so I force it
                                                                         # to recognize it.
    else:
        rotMat = scipy.linalg.expm(hat(w,use_sym)*theta)
    return rotMat

def T(w,th,p,use_sym=True):
    R = rot(w,th,use_sym)
    if use_sym:
        Tmat = sym.Matrix([[R[0,0],R[0,1],R[0,2],p[0]],
                           [R[1,0],R[1,1],R[1,2],p[1]],
                           [R[2,0],R[2,1],R[2,2],p[2]],
                           [     0,     0,     0,   1]])
    else:
        Tmat = np.array([[R[0,0],R[0,1],R[0,2],p[0]],
                         [R[1,0],R[1,1],R[1,2],p[1]],
                         [R[2,0],R[2,1],R[2,2],p[2]],
                         [     0,     0,     0,   1]])
    return Tmat

def pos(T,use_sym=True):
    if use_sym:
        p = sym.Matrix([T[0,3],T[1,3],T[2,3]])
    else:
        p = np.array([T[0,3],T[1,3],T[2,3]])
    return p
    
def animate_project(ta,T):
    # Imports required for animation.
    from plotly.offline import init_notebook_mode, iplot
    from IPython.display import display, HTML
    
    #######################
    # Browser configuration.
    def configure_plotly_browser_state():
      import IPython
      display(IPython.core.display.HTML('''
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
                },
              });
            </script>
            '''))
    configure_plotly_browser_state()
    init_notebook_mode(connected=False)
    
    ###############################################
    # Getting data from angle trajectories and position.
    
    xx1=np.sin(ta[2])+0.125*np.cos(ta[2])
    xx2=0.125*np.cos(ta[2])
    xx3=-0.125*np.cos(ta[2])
    xx4=np.sin(ta[2])-0.125*np.cos(ta[2])
    xx5=np.sin(ta[2]+ta[3])+np.sin(ta[2])+0.125*np.cos(ta[2]+ta[3])
    xx6=np.sin(ta[2])+0.125*np.cos(ta[2]+ta[3])
    xx7=np.sin(ta[2])-0.125*np.cos(ta[2]+ta[3])
    xx8=np.sin(ta[2]+ta[3])+np.sin(ta[2])-0.125*np.cos(ta[2]+ta[3])
    xx9=ta[0]-0.1*np.sin(ta[4])
    xx10=ta[0]+0.1*np.sin(ta[4])-0.1*np.cos(ta[4])
    xx11=ta[0]+0.1*np.sin(ta[4])+0.1*np.cos(ta[4])

    
    yy1=0.125*np.sin(ta[2])-np.cos(ta[2])+3
    yy2=0.125*np.sin(ta[2])+3
    yy3=-0.125*np.sin(ta[2])+3
    yy4=-0.125*np.sin(ta[2])-np.cos(ta[2])+3
    yy5=0.125*np.sin(ta[2]+ta[3])-np.cos(ta[2])-np.cos(ta[2]+ta[3])+3
    yy6=0.125*np.sin(ta[2]+ta[3])-np.cos(ta[2])+3
    yy7=-0.125*np.sin(ta[2]+ta[3])-np.cos(ta[2])+3
    yy8=-0.125*np.sin(ta[2]+ta[3])-np.cos(ta[2])-np.cos(ta[2]+ta[3])+3
    yy9=ta[1]+0.1*np.cos(ta[4])
    yy10=ta[1]-0.1*np.sin(ta[4])-0.1*np.cos(ta[4])
    yy11=ta[1]+0.1*np.sin(ta[4])-0.1*np.cos(ta[4])
    

    N = len(ta[0]) # Need this for specifying length of simulation
    
    ####################################
    # Using these to specify axis limits.
    xm=-11
    xM=17
    ym=-1
    yM=8

    ###########################
    # Defining data dictionary.
    # Trajectories are here.
    # Defining data dictionary.
    # Trajectories are here.
    data=[dict(x=xx1, y=yy1, 
               mode='lines', name='Splits', 
               line=dict(width=2, color='blue')
              )
        ]
    
    
    ################################
    # Preparing simulation layout.
    # Title and axis ranges are here.
    layout=dict(xaxis=dict(range=[xm, xM], autorange=False, zeroline=False,dtick=1),
                yaxis=dict(range=[ym, yM], autorange=False, zeroline=False,scaleanchor = "x",dtick=1),
                title='Triple Pendulum Simulation', 
                hovermode='closest',
                updatemenus= [{'type': 'buttons',
                               'buttons': [{'label': 'Play','method': 'animate',
                                            'args': [None, {'frame': {'duration': T, 'redraw': False}}]},
                                           {'args': [[None], {'frame': {'duration': T, 'redraw': False}, 'mode': 'immediate',
                                            'transition': {'duration': 0}}],'label': 'Pause','method': 'animate'}
                                          ]
                              }]
               )
    
    # Defining the frames of the simulation.
    # This is what draws the lines from
    # joint to joint of the pendulum.
    frames=[dict(data=[dict(x=[xx1[k],xx2[k],xx3[k],xx4[k],xx1[k],None,xx5[k],xx6[k],xx7[k],xx8[k],xx5[k],None,xx9[k],xx10[k],xx11[k],xx9[k],None,16,-10,-10], 
                            y=[yy1[k],yy2[k],yy3[k],yy4[k],yy1[k],None,yy5[k],yy6[k],yy7[k],yy8[k],yy5[k],None,yy9[k],yy10[k],yy11[k],yy9[k],None,0,0,7], 
                            mode='lines',
                            line=dict(color='red', width=2)
                            )
                       ])for k in range(N)]
    
    #######################################
    # Putting it all together and plotting.
    figure1=dict(data=data, layout=layout, frames=frames)          
    iplot(figure1)

m, g, J, w, l,lam = sym.symbols('m g J W L lam')

m=1
g=9.81
J=1
w=0.25
l=1

x = sym.Function('x')(t)
y = sym.Function('y')(t)
th1 = sym.Function(r'\theta_1')(t)
th2 = sym.Function(r'\theta_2')(t)
th3 = sym.Function(r'\theta_3')(t)

q = sym.Matrix([x,y,th1,th2,th3])
qdot = q.diff(t)
qddot = qdot.diff(t)

w = sym.Matrix([0,0,1])

Tw1=sym.simplify(T(w,0,[0,3,0])*T(w,-sym.pi/2,[0,0,0])*T(w,th1,[0,0,0])*T(w,0,[l/2,0,0]))
Tw2=sym.simplify(Tw1*T(w,0,[l/2,0,0])*T(w,th2,[0,0,0])*T(w,0,[l/2,0,0]))
Tw3=sym.simplify(T(w,0,[x,y,0])*T(w,th3,[0,0,0]))
triangleI=sym.simplify(T(w,0,[-0.25,1.5,0]))
springI=sym.simplify(T(w,0,[0,3,0])*T(w,-sym.pi/2,[0,0,0])*T(w,0,[1.5*l,0,0]))

center=sym.Matrix([0,0,0,1])
sp=sym.Matrix([l/2,0,0,1])

p1=Tw1*center
p2=Tw2*center
p3=(Tw3*center)-(triangleI*center)
dp1 = sym.simplify(p1.diff(t))
dp2 = sym.simplify(p2.diff(t))
dp3 = sym.simplify(p3.diff(t))

s1=springI*sp
s2=Tw2*sp

KElin = sym.Matrix(sym.Rational(1,2)*m*(dp1.T*dp1+dp2.T*dp2+dp3.T*dp3))
KErot = sym.Matrix([sym.Rational(1,2)*J*(qdot[2]**2+qdot[3]**2)])+sym.Matrix([sym.Rational(1,2)*0.1*(qdot[4]**2)])
KE = sym.simplify(KElin+KErot)
PE1 = sym.Matrix([m*g*(p1[1]+p2[1])])+sym.Matrix([0.1*g*(p3[1])])
PE2=sym.simplify(sym.Matrix([sym.Rational(1,2)*850*((s1[0]-s2[0])**2)]))
PE= sym.simplify(PE1+PE2)
constraint=y


L = sym.simplify(sym.expand(sym.simplify(KE-PE)))
dLdq = sym.simplify(sym.Matrix([L]).jacobian(q).T)
dLdqdot = sym.simplify(sym.Matrix([L]).jacobian(qdot).T)
ddLdqdot_dt = sym.simplify(dLdqdot.diff(t))
dphidq=sym.simplify(sym.Matrix([constraint]).jacobian(q).T)
d2wdt2=sym.simplify(constraint.diff(t).diff(t))
ELlhs = sym.simplify(ddLdqdot_dt-dLdq-lam*dphidq)
ELrhs = sym.Matrix([0,0,0,0,0])
EL = sym.trigsimp(sym.simplify(sym.Eq(ELlhs,ELrhs)))
con =sym.simplify(sym.Eq(d2wdt2, 0))

vx,vy,w1,w2,w3,ax,ay,a1,a2,a3 = sym.symbols('vx vy w_1 w_2 w_3 ax ay a1 a2 a3')
dyn = sym.Matrix([qdot[0],ax,qdot[1],ay,qdot[2],a1,qdot[3],a2,qdot[4],a3])

def dynamics1 (x):

    if x[0]==-0.25:
      ELS=EL.subs({qdot[0]:x[1],qdot[1]:x[3],qdot[2]:x[5],qdot[3]:x[7],qdot[4]:x[9],q[0]:x[0],q[1]:x[2],q[2]:x[4],q[3]:x[6],q[4]:x[8],qddot[0]:ax,qddot[1]:ay,qddot[2]:a1,qddot[3]:a2,qddot[4]:a3})
      Cons=con.subs({qdot[0]:x[1],qdot[1]:x[3],qdot[2]:x[5],qdot[3]:x[7],qdot[4]:x[9],q[0]:x[0],q[1]:x[2],q[2]:x[4],q[3]:x[6],q[4]:x[8],qddot[0]:ax,qddot[1]:ay,qddot[2]:a1,qddot[3]:a2,qddot[4]:a3})
      Solution =sym.solve([ELS,Cons],[ax,ay,a1,a2,a3,lam])
      dyn_dummy = dyn.subs({qdot[0]:vx,qdot[1]:vy,qdot[2]:w1,qdot[3]:w2,qdot[4]:w3,ax:Solution[ax],ay:Solution[ay],a1:Solution[a1],a2:Solution[a2],a3:Solution[a3]})
      lam_dyn = sym.lambdify([q[0],vx,q[1],vy,q[2],w1,q[3],w2,q[4],w3],dyn_dummy)
    
    else:
      ELS=EL.subs({qdot[0]:x[1],qdot[1]:x[3],qdot[2]:x[5],qdot[3]:x[7],qdot[4]:x[9],q[0]:x[0],q[1]:x[2],q[2]:x[4],q[3]:x[6],q[4]:x[8],qddot[0]:ax,qddot[1]:ay,qddot[2]:a1,qddot[3]:a2,qddot[4]:a3,lam:0})
      Solution =sym.solve([ELS],[ax,ay,a1,a2,a3])
      dyn_dummy = dyn.subs({qdot[0]:vx,qdot[1]:vy,qdot[2]:w1,qdot[3]:w2,qdot[4]:w3,ax:Solution[ax],ay:Solution[ay],a1:Solution[a1],a2:Solution[a2],a3:Solution[a3]})
      lam_dyn = sym.lambdify([q[0],vx,q[1],vy,q[2],w1,q[3],w2,q[4],w3],dyn_dummy)

    
    xdot=lam_dyn(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])
    return np.squeeze(xdot)

l1,eqx,eqy = sym.symbols('lambda1 eqx eqy')

point1=sym.Matrix([0.5,-0.125,0,1])
point2=sym.Matrix([-0.5,-0.125,0,1])

point3=sym.Matrix([0,0.1,0,1])
point4=sym.Matrix([0.1,-0.1,0,1])
point5=sym.Matrix([-0.1,-0.1,0,1])

RP1=Tw2*point1
RP2=Tw2*point2
RP3=Tw3*point3
RP4=Tw3*point4
RP5=Tw3*point5

RPL1=RP1-RP2
RPL2=RP3-RP4

equationR=sym.trigsimp((eqy-RP1[1])/(RPL1[1])-(eqx-RP1[0])/(RPL1[0]))
equationT1=sym.trigsimp((eqy-RP3[1])/(RPL2[1])-(eqx-RP3[0])/(RPL2[0]))
equationT2=sym.trigsimp(eqy-RP4[1])

im1=sym.simplify(equationR.subs({eqx:RP4[0],eqy:RP4[1]}))
im2=sym.simplify(equationR.subs({eqx:RP3[0],eqy:RP3[1]}))
im3=sym.simplify(equationT1.subs({eqx:RP1[0],eqy:RP1[1]}))
im4=sym.simplify(equationT1.subs({eqx:RP2[0],eqy:RP2[1]}))
im5=sym.simplify(equationT2.subs({eqx:RP1[0],eqy:RP1[1]}))
im6=sym.simplify(equationT2.subs({eqx:RP2[0],eqy:RP2[1]}))


lam_im1=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],im1)
lam_im2=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],im2)
lam_im3=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],im3)
lam_im4=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],im4)
lam_im5=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],im5)
lam_im6=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],im6)

dim1dq=sym.simplify(sym.Matrix([im1]).jacobian(q).T)
lam_phiq11=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim1dq[0])
lam_phiq21=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim1dq[1])
lam_phiq31=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim1dq[2])
lam_phiq41=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim1dq[3])
lam_phiq51=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim1dq[4])

dim2dq=sym.simplify(sym.Matrix([im2]).jacobian(q).T)
lam_phiq12=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim2dq[0])
lam_phiq22=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim2dq[1])
lam_phiq32=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim2dq[2])
lam_phiq42=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim2dq[3])
lam_phiq52=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim2dq[4])

dim3dq=sym.simplify(sym.Matrix([im3]).jacobian(q).T)
lam_phiq13=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim3dq[0])
lam_phiq23=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim3dq[1])
lam_phiq33=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim3dq[2])
lam_phiq43=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim3dq[3])
lam_phiq53=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim3dq[4])

dim4dq=sym.simplify(sym.Matrix([im4]).jacobian(q).T)
lam_phiq14=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim4dq[0])
lam_phiq24=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim4dq[1])
lam_phiq34=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim4dq[2])
lam_phiq44=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim4dq[3])
lam_phiq54=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim4dq[4])

dim5dq=sym.simplify(sym.Matrix([im5]).jacobian(q).T)
lam_phiq15=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim5dq[0])
lam_phiq25=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim5dq[1])
lam_phiq35=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim5dq[2])
lam_phiq45=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim5dq[3])
lam_phiq55=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim5dq[4])

dim6dq=sym.simplify(sym.Matrix([im6]).jacobian(q).T)
lam_phiq16=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim6dq[0])
lam_phiq26=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim6dq[1])
lam_phiq36=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim6dq[2])
lam_phiq46=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim6dq[3])
lam_phiq56=sym.lambdify([q[0],q[1],q[2],q[3],q[4]],dim6dq[4])

def phi_condition(x):
  count=0
  if lam_im1(x[0],x[2],x[4],x[6],x[8])< 1e-2:
    count=1
  elif lam_im2(x[0],x[2],x[4],x[6],x[8])< 1e-2:
    count=2
  elif lam_im3(x[0],x[2],x[4],x[6],x[8])< 1e-2:
    count=3
  elif lam_im4(x[0],x[2],x[4],x[6],x[8])< 1e-2:
    count=4
  elif lam_im5(x[0],x[2],x[4],x[6],x[8])< 1e-2:
    count=5
  elif lam_im6(x[0],x[2],x[4],x[6],x[8])< 1e-2:
    count=6
  return count

H=sym.simplify(dLdqdot.dot(qdot)-L[0])
H_dummy = H.subs({qdot[0]:vx,qdot[1]:vy,qdot[2]:w1,qdot[3]:w2,qdot[4]:w3})
lam_H = sym.lambdify([q[0],vx,q[1],vy,q[2],w1,q[3],w2,q[4],w3],H_dummy)
H_plus=H.subs({qdot[0]:ax,qdot[1]:ay,qdot[2]:a1,qdot[3]:a2,qdot[4]:a3})
lam_Hplus = sym.lambdify([q[0],q[1],q[2],q[3],q[4]],H_plus)

Momentum=sym.simplify(dLdqdot)

Momentum_dummy1 = Momentum[0].subs({qdot[0]:vx,qdot[1]:vy,qdot[2]:w1,qdot[3]:w2,qdot[4]:w3})
lam_Momentum1 = sym.lambdify([q[0],vx,q[1],vy,q[2],w1,q[3],w2,q[4],w3],Momentum_dummy1)
Momentum_plus1=Momentum[0].subs({qdot[0]:ax,qdot[1]:ay,qdot[2]:a1,qdot[3]:a2,qdot[4]:a3})
lam_Momentumplus1 = sym.lambdify([q[0],q[1],q[2],q[3],q[4]],Momentum_plus1)

Momentum_dummy2 = Momentum[1].subs({qdot[0]:vx,qdot[1]:vy,qdot[2]:w1,qdot[3]:w2,qdot[4]:w3})
lam_Momentum2 = sym.lambdify([q[0],vx,q[1],vy,q[2],w1,q[3],w2,q[4],w3],Momentum_dummy2)
Momentum_plus2=Momentum[1].subs({qdot[0]:ax,qdot[1]:ay,qdot[2]:a1,qdot[3]:a2,qdot[4]:a3})
lam_Momentumplus2 = sym.lambdify([q[0],q[1],q[2],q[3],q[4]],Momentum_plus2)

Momentum_dummy3 = Momentum[2].subs({qdot[0]:vx,qdot[1]:vy,qdot[2]:w1,qdot[3]:w2,qdot[4]:w3})
lam_Momentum3 = sym.lambdify([q[0],vx,q[1],vy,q[2],w1,q[3],w2,q[4],w3],Momentum_dummy3)
Momentum_plus3=Momentum[2].subs({qdot[0]:ax,qdot[1]:ay,qdot[2]:a1,qdot[3]:a2,qdot[4]:a3})
lam_Momentumplus3 = sym.lambdify([q[0],q[1],q[2],q[3],q[4]],Momentum_plus3)

Momentum_dummy4 = Momentum[3].subs({qdot[0]:vx,qdot[1]:vy,qdot[2]:w1,qdot[3]:w2,qdot[4]:w3})
lam_Momentum4 = sym.lambdify([q[0],vx,q[1],vy,q[2],w1,q[3],w2,q[4],w3],Momentum_dummy3)
Momentum_plus4=Momentum[3].subs({qdot[0]:ax,qdot[1]:ay,qdot[2]:a1,qdot[3]:a2,qdot[4]:a3})
lam_Momentumplus4 = sym.lambdify([q[0],q[1],q[2],q[3],q[4]],Momentum_plus4)

Momentum_dummy5 = Momentum[4].subs({qdot[0]:vx,qdot[1]:vy,qdot[2]:w1,qdot[3]:w2,qdot[4]:w3})
lam_Momentum5 = sym.lambdify([q[0],vx,q[1],vy,q[2],w1,q[3],w2,q[4],w3],Momentum_dummy5)
Momentum_plus5=Momentum[4].subs({qdot[0]:ax,qdot[1]:ay,qdot[2]:a1,qdot[3]:a2,qdot[4]:a3})
lam_Momentumplus5 = sym.lambdify([q[0],q[1],q[2],q[3],q[4]],Momentum_plus5)

def impact_update(x,c):
  print(c)
  if c==1:
    HamiltonianEQs = sym.Eq(lam_Hplus(x[0],x[2],x[4],x[6],x[8])-lam_H(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), 0)
    MomentumEQ1 = sym.Eq(lam_Momentumplus1(x[0],x[2],x[4],x[6],x[8])-lam_Momentum1(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq11(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ2 = sym.Eq(lam_Momentumplus2(x[0],x[2],x[4],x[6],x[8])-lam_Momentum2(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq21(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ3 = sym.Eq(lam_Momentumplus3(x[0],x[2],x[4],x[6],x[8])-lam_Momentum3(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq31(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ4 = sym.Eq(lam_Momentumplus4(x[0],x[2],x[4],x[6],x[8])-lam_Momentum4(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq41(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ5 = sym.Eq(lam_Momentumplus5(x[0],x[2],x[4],x[6],x[8])-lam_Momentum5(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq51(x[0],x[2],x[4],x[6],x[8]))
    EulerSols = sym.solve([HamiltonianEQs, MomentumEQ1,MomentumEQ2,MomentumEQ3,MomentumEQ4,MomentumEQ5],[ax,ay,a1,a2,a3,l1])
    s1=np.real(complex(EulerSols[0][5]))
    s2=np.real(complex(EulerSols[1][5]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols[0][0],EulerSols[0][1],EulerSols[0][2],EulerSols[0][3],EulerSols[0][4]])
    else:
      sol1=sym.Matrix([EulerSols[1][0],EulerSols[1][1],EulerSols[1][2],EulerSols[1][3],EulerSols[1][4]])
  
  elif c==2:
    HamiltonianEQs = sym.Eq(lam_Hplus(x[0],x[2],x[4],x[6],x[8])-lam_H(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), 0)
    MomentumEQ1 = sym.Eq(lam_Momentumplus1(x[0],x[2],x[4],x[6],x[8])-lam_Momentum1(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq12(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ2 = sym.Eq(lam_Momentumplus2(x[0],x[2],x[4],x[6],x[8])-lam_Momentum2(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq22(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ3 = sym.Eq(lam_Momentumplus3(x[0],x[2],x[4],x[6],x[8])-lam_Momentum3(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq32(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ4 = sym.Eq(lam_Momentumplus4(x[0],x[2],x[4],x[6],x[8])-lam_Momentum4(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq42(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ5 = sym.Eq(lam_Momentumplus5(x[0],x[2],x[4],x[6],x[8])-lam_Momentum5(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq52(x[0],x[2],x[4],x[6],x[8]))
    EulerSols = sym.solve([HamiltonianEQs, MomentumEQ1,MomentumEQ2,MomentumEQ3,MomentumEQ4,MomentumEQ5],[ax,ay,a1,a2,a3,l1])
    s1=np.real(complex(EulerSols[0][5]))
    s2=np.real(complex(EulerSols[1][5]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols[0][0],EulerSols[0][1],EulerSols[0][2],EulerSols[0][3],EulerSols[0][4]])
    else:
      sol1=sym.Matrix([EulerSols[1][0],EulerSols[1][1],EulerSols[1][2],EulerSols[1][3],EulerSols[1][4]])
  
  elif c==3:
    HamiltonianEQs = sym.Eq(lam_Hplus(x[0],x[2],x[4],x[6],x[8])-lam_H(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), 0)
    MomentumEQ1 = sym.Eq(lam_Momentumplus1(x[0],x[2],x[4],x[6],x[8])-lam_Momentum1(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq13(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ2 = sym.Eq(lam_Momentumplus2(x[0],x[2],x[4],x[6],x[8])-lam_Momentum2(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq23(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ3 = sym.Eq(lam_Momentumplus3(x[0],x[2],x[4],x[6],x[8])-lam_Momentum3(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq33(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ4 = sym.Eq(lam_Momentumplus4(x[0],x[2],x[4],x[6],x[8])-lam_Momentum4(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq43(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ5 = sym.Eq(lam_Momentumplus5(x[0],x[2],x[4],x[6],x[8])-lam_Momentum5(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq53(x[0],x[2],x[4],x[6],x[8]))
    EulerSols = sym.solve([HamiltonianEQs, MomentumEQ1,MomentumEQ2,MomentumEQ3,MomentumEQ4,MomentumEQ5],[ax,ay,a1,a2,a3,l1])
    s1=np.real(complex(EulerSols[0][5]))
    s2=np.real(complex(EulerSols[1][5]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols[0][0],EulerSols[0][1],EulerSols[0][2],EulerSols[0][3],EulerSols[0][4]])
    else:
      sol1=sym.Matrix([EulerSols[1][0],EulerSols[1][1],EulerSols[1][2],EulerSols[1][3],EulerSols[1][4]])

  elif c==4:
    HamiltonianEQs = sym.Eq(lam_Hplus(x[0],x[2],x[4],x[6],x[8])-lam_H(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), 0)
    MomentumEQ1 = sym.Eq(lam_Momentumplus1(x[0],x[2],x[4],x[6],x[8])-lam_Momentum1(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq14(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ2 = sym.Eq(lam_Momentumplus2(x[0],x[2],x[4],x[6],x[8])-lam_Momentum2(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq24(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ3 = sym.Eq(lam_Momentumplus3(x[0],x[2],x[4],x[6],x[8])-lam_Momentum3(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq34(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ4 = sym.Eq(lam_Momentumplus4(x[0],x[2],x[4],x[6],x[8])-lam_Momentum4(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq44(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ5 = sym.Eq(lam_Momentumplus5(x[0],x[2],x[4],x[6],x[8])-lam_Momentum5(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq54(x[0],x[2],x[4],x[6],x[8]))
    EulerSols = sym.solve([HamiltonianEQs, MomentumEQ1,MomentumEQ2,MomentumEQ3,MomentumEQ4,MomentumEQ5],[ax,ay,a1,a2,a3,l1])
    s1=np.real(complex(EulerSols[0][5]))
    s2=np.real(complex(EulerSols[1][5]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols[0][0],EulerSols[0][1],EulerSols[0][2],EulerSols[0][3],EulerSols[0][4]])
    else:
      sol1=sym.Matrix([EulerSols[1][0],EulerSols[1][1],EulerSols[1][2],EulerSols[1][3],EulerSols[1][4]])

  elif c==5:
    HamiltonianEQs = sym.Eq(lam_Hplus(x[0],x[2],x[4],x[6],x[8])-lam_H(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), 0)
    MomentumEQ1 = sym.Eq(lam_Momentumplus1(x[0],x[2],x[4],x[6],x[8])-lam_Momentum1(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq15(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ2 = sym.Eq(lam_Momentumplus2(x[0],x[2],x[4],x[6],x[8])-lam_Momentum2(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq25(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ3 = sym.Eq(lam_Momentumplus3(x[0],x[2],x[4],x[6],x[8])-lam_Momentum3(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq35(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ4 = sym.Eq(lam_Momentumplus4(x[0],x[2],x[4],x[6],x[8])-lam_Momentum4(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq45(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ5 = sym.Eq(lam_Momentumplus5(x[0],x[2],x[4],x[6],x[8])-lam_Momentum5(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq55(x[0],x[2],x[4],x[6],x[8]))
    EulerSols = sym.solve([HamiltonianEQs, MomentumEQ1,MomentumEQ2,MomentumEQ3,MomentumEQ4,MomentumEQ5],[ax,ay,a1,a2,a3,l1])
    s1=np.real(complex(EulerSols[0][5]))
    s2=np.real(complex(EulerSols[1][5]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols[0][0],EulerSols[0][1],EulerSols[0][2],EulerSols[0][3],EulerSols[0][4]])
    else:
      sol1=sym.Matrix([EulerSols[1][0],EulerSols[1][1],EulerSols[1][2],EulerSols[1][3],EulerSols[1][4]])

  elif c==6:
    HamiltonianEQs = sym.Eq(lam_Hplus(x[0],x[2],x[4],x[6],x[8])-lam_H(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), 0)
    MomentumEQ1 = sym.Eq(lam_Momentumplus1(x[0],x[2],x[4],x[6],x[8])-lam_Momentum1(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq16(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ2 = sym.Eq(lam_Momentumplus2(x[0],x[2],x[4],x[6],x[8])-lam_Momentum2(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq26(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ3 = sym.Eq(lam_Momentumplus3(x[0],x[2],x[4],x[6],x[8])-lam_Momentum3(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq36(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ4 = sym.Eq(lam_Momentumplus4(x[0],x[2],x[4],x[6],x[8])-lam_Momentum4(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq46(x[0],x[2],x[4],x[6],x[8]))
    MomentumEQ5 = sym.Eq(lam_Momentumplus5(x[0],x[2],x[4],x[6],x[8])-lam_Momentum5(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]), l1*lam_phiq56(x[0],x[2],x[4],x[6],x[8]))
    EulerSols = sym.solve([HamiltonianEQs, MomentumEQ1,MomentumEQ2,MomentumEQ3,MomentumEQ4,MomentumEQ5],[ax,ay,a1,a2,a3,l1])
    s1=np.real(complex(EulerSols[0][5]))
    s2=np.real(complex(EulerSols[1][5]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols[0][0],EulerSols[0][1],EulerSols[0][2],EulerSols[0][3],EulerSols[0][4]])
    else:
      sol1=sym.Matrix([EulerSols[1][0],EulerSols[1][1],EulerSols[1][2],EulerSols[1][3],EulerSols[1][4]])
  
  newx=np.copy(x)
  newx[1]=np.real(complex(sol1[0]))
  newx[3]=np.real(complex(sol1[1]))
  newx[5]=np.real(complex(sol1[2]))
  newx[7]=np.real(complex(sol1[3]))
  newx[9]=np.real(complex(sol1[4]))
  print(newx[1])
  print(newx[3])
  print(newx[9])
  return newx

x0 = np.array([-0.25,0,1.5,0,np.pi/2,0,0,0,0,0])
dt=0.01
x1vec,N,time=simulate_impacts1(dynamics1,x0,dt,phi_condition,impact_update)
print(time)

m, g, J = sym.symbols('m g J')

m=1
g=9.81
J=1

x1 = sym.Function('x')(t)
y1 = sym.Function('y')(t)
th11 = sym.Function(r'\theta_1')(t)

q1 = sym.Matrix([x1,y1,th11])
qdot1 = q1.diff(t)
qddot1 = qdot1.diff(t)

w1 = sym.Matrix([0,0,1])
Tri=sym.simplify(T(w1,0,[x1,y1,0])*T(w1,th11,[0,0,0]))
center1=sym.Matrix([0,0,0,1])

p11=Tri*center1
dp11 = sym.simplify(p11.diff(t))

KElin1 = sym.Matrix(sym.Rational(1,2)*m*(dp11.T*dp11))
KErot1 = sym.Matrix([sym.Rational(1,2)*J*(qdot1[2]**2)])
KE1 = sym.simplify(KElin1+KErot1)
PET = sym.simplify(sym.Matrix([m*g*(p11[1])]))

L1 = sym.simplify(sym.expand(sym.simplify(KE1-PET)))
dLdq1 = sym.simplify(sym.Matrix([L1]).jacobian(q1).T)
dLdqdot1 = sym.simplify(sym.Matrix([L1]).jacobian(qdot1).T)
ddLdqdot_dt1 = sym.simplify(dLdqdot1.diff(t))
ELlhs1 = sym.simplify(ddLdqdot_dt1-dLdq1)
ELrhs1 = sym.Matrix([0,0,0])
EL1 = sym.trigsimp(sym.simplify(sym.Eq(ELlhs1,ELrhs1)))

EulerSols = sym.solve(EL1,[qddot1[0],qddot1[1],qddot1[2]])

vx1, vy1, w11 = sym.symbols('vx1 vy1 w11')
dyn1= sym.Matrix([qdot1[0],EulerSols[qddot1[0]],qdot1[1],EulerSols[qddot1[1]],qdot1[2],EulerSols[qddot1[2]]])
dyn_dummy1 = dyn1.subs({qdot1[0]:vx1,qdot1[1]:vy1,qdot1[2]:w11})
lam_dyn1 = sym.lambdify([q1[0],vx1,q1[1],vy1,q1[2],w11],dyn_dummy1)

def dynamics (x):
      xdot=lam_dyn1(x[0],x[1],x[2],x[3],x[4],x[5])
      return np.squeeze(xdot)
      
lamb1,eqx1,eqy1,ax1,ay1,aw11 = sym.symbols('lamb1 eqx1 eqy1 ax1,ay1 aw11')

point3=sym.Matrix([0,0.1,0,1])
point4=sym.Matrix([0.1,-0.1,0,1])
point5=sym.Matrix([-0.1,-0.1,0,1])

RA=Tri*point3
RB=Tri*point4
RC=Tri*point5

eqground=eqy1
eqwall=eqx1+10

imAG=sym.simplify(eqground.subs({eqx1:RA[0],eqy1:RA[1]}))
imBG=sym.simplify(eqground.subs({eqx1:RB[0],eqy1:RB[1]}))
imCG=sym.simplify(eqground.subs({eqx1:RB[0],eqy1:RB[1]}))
imAW=sym.simplify(eqwall.subs({eqx1:RA[0],eqy1:RA[1]}))
imBW=sym.simplify(eqwall.subs({eqx1:RB[0],eqy1:RB[1]}))
imCW=sym.simplify(eqwall.subs({eqx1:RB[0],eqy1:RB[1]}))

lam_imAG=sym.lambdify([q1[0],q1[1],q1[2]],imAG)
lam_imBG=sym.lambdify([q1[0],q1[1],q1[2]],imBG)
lam_imCG=sym.lambdify([q1[0],q1[1],q1[2]],imCG)
lam_imAW=sym.lambdify([q1[0],q1[1],q1[2]],imAW)
lam_imBW=sym.lambdify([q1[0],q1[1],q1[2]],imBW)
lam_imCW=sym.lambdify([q1[0],q1[1],q1[2]],imCW)

dimAGdq=sym.simplify(sym.Matrix([imAG]).jacobian(q1).T)
lam_phiq1AG=sym.lambdify([q1[0],q1[1],q1[2]],dimAGdq[0])
lam_phiq2AG=sym.lambdify([q1[0],q1[1],q1[2]],dimAGdq[1])
lam_phiq3AG=sym.lambdify([q1[0],q1[1],q1[2]],dimAGdq[2])

dimBGdq=sym.simplify(sym.Matrix([imBG]).jacobian(q1).T)
lam_phiq1BG=sym.lambdify([q1[0],q1[1],q1[2]],dimBGdq[0])
lam_phiq2BG=sym.lambdify([q1[0],q1[1],q1[2]],dimBGdq[1])
lam_phiq3BG=sym.lambdify([q1[0],q1[1],q1[2]],dimBGdq[2])

dimCGdq=sym.simplify(sym.Matrix([imCG]).jacobian(q1).T)
lam_phiq1CG=sym.lambdify([q1[0],q1[1],q1[2]],dimCGdq[0])
lam_phiq2CG=sym.lambdify([q1[0],q1[1],q1[2]],dimCGdq[1])
lam_phiq3CG=sym.lambdify([q1[0],q1[1],q1[2]],dimCGdq[2])

dimAWdq=sym.simplify(sym.Matrix([imAW]).jacobian(q1).T)
lam_phiq1AW=sym.lambdify([q1[0],q1[1],q1[2]],dimAWdq[0])
lam_phiq2AW=sym.lambdify([q1[0],q1[1],q1[2]],dimAWdq[1])
lam_phiq3AW=sym.lambdify([q1[0],q1[1],q1[2]],dimAWdq[2])

dimBWdq=sym.simplify(sym.Matrix([imBW]).jacobian(q1).T)
lam_phiq1BW=sym.lambdify([q1[0],q1[1],q1[2]],dimBWdq[0])
lam_phiq2BW=sym.lambdify([q1[0],q1[1],q1[2]],dimBWdq[1])
lam_phiq3BW=sym.lambdify([q1[0],q1[1],q1[2]],dimBWdq[2])

dimCWdq=sym.simplify(sym.Matrix([imCW]).jacobian(q1).T)
lam_phiq1CW=sym.lambdify([q1[0],q1[1],q1[2]],dimCWdq[0])
lam_phiq2CW=sym.lambdify([q1[0],q1[1],q1[2]],dimCWdq[1])
lam_phiq3CW=sym.lambdify([q1[0],q1[1],q1[2]],dimCWdq[2])

def phi_condition1(x):
  count=0
  if lam_imAG(x[0],x[2],x[4])< 1e-2:
    count=1
  elif lam_imBG(x[0],x[2],x[4])< 1e-2:
    count=2
  elif lam_imCG(x[0],x[2],x[4])< 1e-2:
    count=3
  elif lam_imAW(x[0],x[2],x[4])< 1e-2:
    count=4
  elif lam_imBW(x[0],x[2],x[4])< 1e-2:
    count=5
  elif lam_imCW(x[0],x[2],x[4])< 1e-2:
    count=6
  return count

H1=sym.simplify(dLdqdot1.dot(qdot1)-L1[0])
H_dummy1 = H1.subs({qdot1[0]:vx1,qdot1[1]:vy1,qdot1[2]:w11})
lam_H1 = sym.lambdify([q1[0],vx1,q1[1],vy1,q1[2],w11],H_dummy1)
H_plus1=H1.subs({qdot1[0]:ax1,qdot1[1]:ay1,qdot1[2]:aw11})
lam_Hplus1 = sym.lambdify([q1[0],q1[1],q1[2]],H_plus1)

Momentum1=sym.simplify(dLdqdot1)

Momentum_dummy11 = Momentum1[0].subs({qdot1[0]:vx1,qdot1[1]:vy1,qdot1[2]:w11})
lam_Momentum11 = sym.lambdify([q1[0],vx1,q1[1],vy1,q1[2],w11],Momentum_dummy11)
Momentum_plus11=Momentum1[0].subs({qdot1[0]:ax1,qdot1[1]:ay1,qdot1[2]:aw11})
lam_Momentumplus11 = sym.lambdify([q1[0],q1[1],q1[2]],Momentum_plus11)

Momentum_dummy21 = Momentum1[1].subs({qdot1[0]:vx1,qdot1[1]:vy1,qdot1[2]:w11})
lam_Momentum21 = sym.lambdify([q1[0],vx1,q1[1],vy1,q1[2],w11],Momentum_dummy21)
Momentum_plus21=Momentum1[1].subs({qdot1[0]:ax1,qdot1[1]:ay1,qdot1[2]:aw11})
lam_Momentumplus21 = sym.lambdify([q1[0],q1[1],q1[2]],Momentum_plus21)

Momentum_dummy31 = Momentum1[2].subs({qdot1[0]:vx1,qdot1[1]:vy1,qdot1[2]:w11})
lam_Momentum31 = sym.lambdify([q1[0],vx1,q1[1],vy1,q1[2],w11],Momentum_dummy31)
Momentum_plus31=Momentum1[2].subs({qdot1[0]:ax1,qdot1[1]:ay1,qdot1[2]:aw11})
lam_Momentumplus31 = sym.lambdify([q1[0],q1[1],q1[2]],Momentum_plus31)

def impact_update1(x,c):
  x[1]=0.8*x[1]
  x[3]=0.8*x[3]
  x[5]=0.8*x[5]
  
  if c==1:
    HamiltonianEQs1 = sym.Eq(lam_Hplus1(x[0],x[2],x[4])-lam_H1(x[0],x[1],x[2],x[3],x[4],x[5]), 0)
    MomentumEQ11 = sym.Eq(lam_Momentumplus11(x[0],x[2],x[4])-lam_Momentum11(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq1AG(x[0],x[2],x[4]))
    MomentumEQ21 = sym.Eq(lam_Momentumplus21(x[0],x[2],x[4])-lam_Momentum21(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq2AG(x[0],x[2],x[4]))
    MomentumEQ31 = sym.Eq(lam_Momentumplus31(x[0],x[2],x[4])-lam_Momentum31(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq3AG(x[0],x[2],x[4]))
    EulerSols1 = sym.solve([HamiltonianEQs1, MomentumEQ11,MomentumEQ21,MomentumEQ31],[ax1,ay1,aw11,lamb1])
    s1=np.real(complex(EulerSols1[0][3]))
    s2=np.real(complex(EulerSols1[1][3]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols1[0][0],EulerSols1[0][1],EulerSols1[0][2]])
    else:
      sol1=sym.Matrix([EulerSols1[1][0],EulerSols1[1][1],EulerSols1[1][2]])
  
  elif c==2:
    HamiltonianEQs1 = sym.Eq(lam_Hplus1(x[0],x[2],x[4])-lam_H1(x[0],x[1],x[2],x[3],x[4],x[5]), 0)
    MomentumEQ11 = sym.Eq(lam_Momentumplus11(x[0],x[2],x[4])-lam_Momentum11(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq1BG(x[0],x[2],x[4]))
    MomentumEQ21 = sym.Eq(lam_Momentumplus21(x[0],x[2],x[4])-lam_Momentum21(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq2BG(x[0],x[2],x[4]))
    MomentumEQ31 = sym.Eq(lam_Momentumplus31(x[0],x[2],x[4])-lam_Momentum31(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq3BG(x[0],x[2],x[4]))
    EulerSols1 = sym.solve([HamiltonianEQs1, MomentumEQ11,MomentumEQ21,MomentumEQ31],[ax1,ay1,aw11,lamb1])
    s1=np.real(complex(EulerSols1[0][3]))
    s2=np.real(complex(EulerSols1[1][3]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols1[0][0],EulerSols1[0][1],EulerSols1[0][2]])
    else:
      sol1=sym.Matrix([EulerSols1[1][0],EulerSols1[1][1],EulerSols1[1][2]])
  
  elif c==3:
    HamiltonianEQs1 = sym.Eq(lam_Hplus1(x[0],x[2],x[4])-lam_H1(x[0],x[1],x[2],x[3],x[4],x[5]), 0)
    MomentumEQ11 = sym.Eq(lam_Momentumplus11(x[0],x[2],x[4])-lam_Momentum11(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq1CG(x[0],x[2],x[4]))
    MomentumEQ21 = sym.Eq(lam_Momentumplus21(x[0],x[2],x[4])-lam_Momentum21(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq2CG(x[0],x[2],x[4]))
    MomentumEQ31 = sym.Eq(lam_Momentumplus31(x[0],x[2],x[4])-lam_Momentum31(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq3CG(x[0],x[2],x[4]))
    EulerSols1 = sym.solve([HamiltonianEQs1, MomentumEQ11,MomentumEQ21,MomentumEQ31],[ax1,ay1,aw11,lamb1])
    s1=np.real(complex(EulerSols1[0][3]))
    s2=np.real(complex(EulerSols1[1][3]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols1[0][0],EulerSols1[0][1],EulerSols1[0][2]])
    else:
      sol1=sym.Matrix([EulerSols1[1][0],EulerSols1[1][1],EulerSols1[1][2]])

  elif c==4:
    HamiltonianEQs1 = sym.Eq(lam_Hplus1(x[0],x[2],x[4])-lam_H1(x[0],x[1],x[2],x[3],x[4],x[5]), 0)
    MomentumEQ11 = sym.Eq(lam_Momentumplus11(x[0],x[2],x[4])-lam_Momentum11(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq1AW(x[0],x[2],x[4]))
    MomentumEQ21 = sym.Eq(lam_Momentumplus21(x[0],x[2],x[4])-lam_Momentum21(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq2AW(x[0],x[2],x[4]))
    MomentumEQ31 = sym.Eq(lam_Momentumplus31(x[0],x[2],x[4])-lam_Momentum31(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq3AW(x[0],x[2],x[4]))
    EulerSols1 = sym.solve([HamiltonianEQs1, MomentumEQ11,MomentumEQ21,MomentumEQ31],[ax1,ay1,aw11,lamb1])
    s1=np.real(complex(EulerSols1[0][3]))
    s2=np.real(complex(EulerSols1[1][3]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols1[0][0],EulerSols1[0][1],EulerSols1[0][2]])
    else:
      sol1=sym.Matrix([EulerSols1[1][0],EulerSols1[1][1],EulerSols1[1][2]])

  elif c==5:
    HamiltonianEQs1 = sym.Eq(lam_Hplus1(x[0],x[2],x[4])-lam_H1(x[0],x[1],x[2],x[3],x[4],x[5]), 0)
    MomentumEQ11 = sym.Eq(lam_Momentumplus11(x[0],x[2],x[4])-lam_Momentum11(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq1BW(x[0],x[2],x[4]))
    MomentumEQ21 = sym.Eq(lam_Momentumplus21(x[0],x[2],x[4])-lam_Momentum21(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq2BW(x[0],x[2],x[4]))
    MomentumEQ31 = sym.Eq(lam_Momentumplus31(x[0],x[2],x[4])-lam_Momentum31(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq3BW(x[0],x[2],x[4]))
    EulerSols1 = sym.solve([HamiltonianEQs1, MomentumEQ11,MomentumEQ21,MomentumEQ31],[ax1,ay1,aw11,lamb1])
    s1=np.real(complex(EulerSols1[0][3]))
    s2=np.real(complex(EulerSols1[1][3]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols1[0][0],EulerSols1[0][1],EulerSols1[0][2]])
    else:
      sol1=sym.Matrix([EulerSols1[1][0],EulerSols1[1][1],EulerSols1[1][2]])

  elif c==6:
    HamiltonianEQs1 = sym.Eq(lam_Hplus1(x[0],x[2],x[4])-lam_H1(x[0],x[1],x[2],x[3],x[4],x[5]), 0)
    MomentumEQ11 = sym.Eq(lam_Momentumplus11(x[0],x[2],x[4])-lam_Momentum11(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq1CW(x[0],x[2],x[4]))
    MomentumEQ21 = sym.Eq(lam_Momentumplus21(x[0],x[2],x[4])-lam_Momentum21(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq2CW(x[0],x[2],x[4]))
    MomentumEQ31 = sym.Eq(lam_Momentumplus31(x[0],x[2],x[4])-lam_Momentum31(x[0],x[1],x[2],x[3],x[4],x[5]), lamb1*lam_phiq3CW(x[0],x[2],x[4]))
    EulerSols1 = sym.solve([HamiltonianEQs1, MomentumEQ11,MomentumEQ21,MomentumEQ31],[ax1,ay1,aw11,lamb1])
    s1=np.real(complex(EulerSols1[0][3]))
    s2=np.real(complex(EulerSols1[1][3]))
    if s1>s2:
      sol1=sym.Matrix([EulerSols1[0][0],EulerSols1[0][1],EulerSols1[0][2]])
    else:
      sol1=sym.Matrix([EulerSols1[1][0],EulerSols1[1][1],EulerSols1[1][2]])
  
  newx=np.copy(x)
  newx[1]=np.real(complex(sol1[0]))
  newx[3]=np.real(complex(sol1[1]))
  newx[5]=np.real(complex(sol1[2]))
  return newx
  
tspan11 = [0,6]
dt11 = 0.01
x011 = np.array([-0.25,x1vec[1][N-1],1.5,x1vec[3][N-1],0,x1vec[9][N-1]])
N11 = int((max(tspan11)-min(tspan11))/dt11)
tvec11 = np.linspace(min(tspan11),max(tspan11),N11)

xvec111 = simulate_impacts(dynamics,x011,tspan11,dt11,phi_condition1,impact_update1)

xfinal=np.hstack((x1vec[0],xvec111[0]))
yfinal=np.hstack((x1vec[2],xvec111[2]))
th3final=np.hstack((x1vec[8],xvec111[4]))

th1i=x1vec[4][N-1]
th2i=x1vec[6][N-1]
th1f=np.pi/2
th2f=0
th1afterimpact=np.zeros(N11)
th2afterimpact=np.zeros(N11)

for i in range(100):
  th1afterimpact[i]=th1i+((th1f-th1i)/100)*i
  th2afterimpact[i]=th2i+((th2f-th2i)/100)*i

for i in range(100,N11):
  th1afterimpact[i]=th1f
  th2afterimpact[i]=th2f

th1final=np.hstack((x1vec[4],th1afterimpact))
th2final=np.hstack((x1vec[6],th2afterimpact))

N111=len(xfinal)
thetaarray33=np.zeros((5,N111))
thetaarray33[0,0:N111]=xfinal
thetaarray33[1,0:N111]=yfinal
thetaarray33[4,0:N111]=th3final
thetaarray33[2,0:N111]=th1final
thetaarray33[3,0:N111]=th2final

animate_project(thetaarray33,5.19)