'''     ________________________________
     
          ASSIGNMENT-7 [EE2703]
               EE19B033
            KORRA SRIKANTH
    _________________________________________
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import sympy as sy


# lowpass function
def lowpass(R1,R2,C1,C2,G,Vi):
    s=sy.symbols('s')
    A=sy.Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0], [0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b=sy.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return (A,b,V)


#converting Vo from sympy.core to sp.lti        
def convert(Vs) :
    Vs = sy.simplify(Vs)    
    num,den = sy.fraction(Vs)
    num = np.array(sy.Poly(num,s).all_coeffs(),dtype = np.complex64)
    den = np.array(sy.Poly(den,s).all_coeffs(),dtype = np.complex64)
    Vo_sig = sp.lti(num,den)
    return Vo_sig     



A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
Vo=V[3]
w=np.logspace(0,8,801)
ss=1j*w
s = sy.symbols('s')
hf=sy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.title(r'Magnitude plot of low pass')
plt.grid(True)
plt.show() 


Vo_signal = convert(Vo)
#calculates the step response of the lowpass
t,unit_res = sp.step(Vo_signal,None,np.linspace(0,100e-6,200001),None)
plt.plot(t,unit_res)
plt.title('unit step response low pass')
plt.show()


#highpass Function
def highpass(R1,R2,C1,C2,G,Vi):
    s=sy.symbols('s')
    A=sy.Matrix([[0,0,1,-1/G],[-(s*R2*C2)/(1+s*R2*C2),1,0,0], [0,-G,G,1],[-1/R1-s*C2-s*C1,s*C2,0,1/R1]])
    b=sy.Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return (A,b,V)
      

#Highpass Filter
A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)
Vo=V[3]
w=np.logspace(0,8,801)
ss=1j*w
s = sy.symbols('s')
hf=sy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.title(r'Magnitude plot of High pass')
plt.grid(True)
plt.show() 


Vo_signal = convert(Vo)
# computes the sinusoidal response of the highpass
t = np.linspace(0,6/(1e6),2001)
Si  = np.sin(2000*np.pi*t)+np.cos(2*1e6*np.pi*t)
t,sin_res,svec = sp.lsim(Vo_signal,Si,t)


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey = True)
axes[0].plot(t,sin_res)
axes[0].grid()
axes[0].set_title(' Output of highpass filter to given sinusoidal Input  ')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('y(t)')

axes[1].plot(t,Si)
axes[1].grid()
axes[1].set_title(' Input to highpass filter')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('x(t)')
plt.show()



t = np.linspace(0.0,3e-5,100001)     
Di = np.exp(-1e5*t)*np.cos(2*1e6*np.pi*t) 
t,sin_res,svec = sp.lsim(Vo_signal,Di,t)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey = True)
axes[0].plot(t,sin_res)
axes[0].grid()
axes[0].set_title('Output of highpass filter to given damping input')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('y(t)')

axes[1].plot(t,Di)
axes[1].grid()
axes[1].set_title('Input to highpass filter')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('x(t)')
plt.show()



# The below code computes the unit step response by passing the Vi as 1/s 
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo=V[3]
Vo_signal = convert(Vo)
t,Vo_time = sp.impulse(Vo_signal,None,np.linspace(0,100e-6,20001))
plt.plot(t,Vo_time)
plt.grid()
plt.title('step response of lowpass(1/s)')
plt.plot()
plt.show()


A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo=V[3]
Vo_signal = convert(Vo) 
t,Vo_time = sp.impulse(Vo_signal,None,np.linspace(0,100e-6,20001))
plt.plot(t,Vo_time)
plt.grid()
plt.title('step response of Highpass(1/s)')
plt.plot()
plt.show()

