
''' ______________________________________

         ASSIGNMENT - 6B [EE2703]
               EE19B033
            KORRA SRIKANTH
       ____________________________
'''
  
from numpy import *
import scipy.signal as sp
from matplotlib.pyplot import *


# Question : 1,2
def ResX(a, w):
    XsNum = np.poly1d([1, a])
    XsDen = np.polymul([1, 2*a, a**2 + 2.25],[1,0,2.25])
    Xs = sp.lti(XsNum, XsDen)
    t, X = sp.impulse(Xs, None, np.linspace(0, 50, 500))
    return t, X


t1, x1 = ResX(0.5, 1.5)
t2, x2 = ResX(0.05, 1.5)

# Plotting the time response for two different values of decay constant
figure(1)
xlabel(r'$t$')
ylabel(r'$x(t)$')
plot(t2,x2,'b',label = 'Damping coeff = 0.5')
plot(t1,x1,'m',label = 'Damping coeff = 0.05')
title(r"System with Decay constant =0.5 & 0.05")
legend()
show()


# Question : 3
def f(t,w):
    return (cos(w*t))*exp(-0.05*t)

# Simulating it for different values of 'w'
figure(2) 
xlabel(r'$t$')
ylabel(r'$x(t)$')
k1 = 0
for i in range(5):
    k = 1.4 + 0.05*k1
    H = sp.lti([1],[1,0,2.25])
    t=linspace(0,50,500)
    u=f(t,k)
    t,y,svec=sp.lsim(H,u,t)
    k1 = k1 + 1
    plot(t,y,label = 'w =  {:.2f}'.format(k))
xlabel(r"$t \to$")
ylabel(r"$x(t) \to$")
title(r"Response of LTI system to various frequencies")
legend()
show()

# Question : 4
# Defining the transfer function of one spring
H1 = sp.lti([1,0,2],[1,0,3,0]) 
tt1,X1 = sp.impulse(H1,None,linspace(0,20,500))

# Defining the transfer function of other spring
H2 = sp.lti([2],[1,0,3,0]) 
tt2,Y1 = sp.impulse(H2,None,linspace(0,20,500))

# Plotting the time Response of coupled spring system
figure(3)
xlabel(r'$t---->$')
plot(tt2,Y1,'r',label = 'y(t)')
plot(tt1,X1,'g',label = 'x(t)')
title(r"Coupled System Response ")
legend()
show()


# Question : 5
tx = linspace(0,500 ,1000 )
Hs_num = poly1d([1])
Hs_den = poly1d([10**(-12),10**(-4),1] )
HH = sp.lti(Hs_num,Hs_den) # Defining the transfer function of two port network
w,S,vs = HH.bode() # Bode 

figure(4)
subplot(2,1,1)
semilogx(w, S)
xlabel(r"$\omega \ \to$")
ylabel(r"$\|H(jw)\|\ (in\ dB)$")
title("Bode plot of the given RLC network")

subplot(2,1,2)
xlabel(r"$\omega \ \to$")
ylabel(r"$\angle H(jw)\ (in\ ^o)$")
semilogx(w, vs)
show()


# Question : 6
tx1 = linspace(0,10**(-2),10**6)
v = cos(10**(3)*tx1) - cos(10**(6)*tx1) # Input signal
t3,v1,svec = sp.lsim(HH,v,tx1)

figure(5)
# Plotting the Steady State Response of output signal
# Plotting the output signal (Zoomed in)
plot(t3,v1,'b',label ='Vo(t)') 
title("Steady state Response")
ylabel(r"$V_{o}(t)   --->$",size=15)
xlabel(r"$t  --->$",size=15)
legend()
show()

tx2 = linspace(0,30*10**(-6),3001)
v = cos(10**(3)*tx2) - cos(10**(6)*tx2) # Input signal
t4,v2,svec = sp.lsim(HH,v,tx2)

# Plotting the Transient Response of output signal
figure(6)
plot(t4,v2,'b',label ='Vo(t)') 
title("Transient Response")
ylabel(r"$V_{o}(t)--->$",size=15)
xlabel(r"$t--->$",size=15)
legend()
show()


