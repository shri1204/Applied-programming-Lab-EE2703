
'''   ___________________________________

             ASSIGNMENT 5 [EE2703]
                  EE19B033
               KORRA SRIKANTH
          ___________________________
'''

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.integrate import *

periodic_e =  lambda x : np.e**(np.remainder(x,2*pi))
e = lambda x : np.e**x
f = lambda x : np.cos(np.cos(x))

#Question: 1
#1200 points in interval of -2pi to 4pi
x0 = linspace(-2*pi,4*pi,1200) 
y0 = list(map(f, x0))  
y1 = list(map(e, x0))

plt.figure(1)
plt.grid(True)
plt.semilogy(x0,y0,color = 'green')
plt.semilogy(x0,cos(cos(x0)),color='orange')
plt.title("$\cos(\cos(x))$ and its periodically extended version")
plt.ylabel(r"$y$ ")
plt.xlabel(r"$x$ (linear)")
plt.legend([r"$\cos(\cos(x))$",r"periodic version of $\cos(\cos(x))$"], loc=1)
plt.show()

plt.figure(2)      
plt.grid(True)
plt.semilogy(x0,y1)
plt.semilogy(x0,periodic_e(x0),color='orange')
plt.title("$e^x$ and its periodically extended version")
plt.ylabel(r"$y$ (log)")
plt.xlabel(r"$x$ (linear)")
plt.legend([r"$e^x$",r"periodic version of $e^x$"], loc='upper right')
plt.show()

#Question: 2
u1 = lambda x,k : e(x)*np.cos(k*x)
v1 = lambda x,k : e(x)*np.sin(k*x)
u2 = lambda x,k : f(x)*np.cos(k*x)
v2 = lambda x,k : f(x)*np.cos(k*x)

# Initializing lists for storing fourier Coeffs
a1 = [0]*26 
b1 = [0]*26
a2 = [0]*26
b2 = [0]*26
a1[0] = quad(u1,0,2*pi,args=(0))  
a2[0] = quad(u2,0,2*pi,args=(0))
F1 = [(1/(2*pi))*a1[0][0]]
F2 = [(1/(2*pi))*a2[0][0]]
for i in range(1,26):
    a1[i] = quad(u1,0,2*pi,args=(i))
    F1.append((1/(pi))*a1[i][0])
    b1[i] = quad(v1,0,2*pi,args=(i))
    F1.append((1/(pi))*b1[i][0])
    a2[i] = quad(u2,0,2*pi,args=(i))
    F2.append((1/(pi))*a2[i][0])
    b2[i] = quad(v2,0,2*pi,args=(i))
    F2.append((1/(pi))*b2[i][0])
x2 = [0]
for i in range(1,26):
    x2.append(i)
    x2.append(i)

#Question: 4
# Calculating coefficients using lstsq method
x = linspace(0,2*pi,401) 
x = x[:-1]
a = zeros((400,51)) # Initializing empty matrix
a[:,0] = 1 # Making first column 1
for i in range(1,26):
    a[:,2*i-1] = cos(i*x) # Constructing the required matrix
    a[:,2*i] = sin(i*x)

#Question: 5
# Solving for the coefficients which give least square error
Y = linalg.lstsq(a,e(x),rcond=-1)[0] 
Z = linalg.lstsq(a,f(x),rcond=-1)[0]

#Question: 3,6
# plotting coefficients of e(x) in semilogy and loglog
plt.figure(3) 
plt.title('Function: exp(x) coeffs vs n semilogy')
plt.xlabel(r'$n$')
plt.ylabel(r'$\ exp(x) \ Coeffs$')
plt.semilogy(x2,F1,'ro',label = 'Fourier series')
plt.semilogy(x2,Y,'go',label= 'least square')
plt.legend()
plt.grid()
plt.show()

plt.figure(4)
plt.title('Function: exp(x) coeffs vs n loglog')
plt.xlabel(r'$n$')
plt.ylabel(r'$\ exp(x) \ Coeffs$')
plt.loglog(x2,F1,'ro',label = 'Fourier series')
plt.loglog(x2,Y,'go',label = 'least square')
plt.legend()
plt.grid()
plt.show()

# plotting coefficients of cos(cos(x)) in semilogy and loglog
plt.figure(5) 
plt.title('Function: cos(cos(x)) coeffs vs n semilogy')
plt.xlabel(r'$n$')
plt.ylabel(r'$ \ cos(cos(x) \ Coeffs$')
plt.semilogy(x2,F2,'ro',label = 'Fourier series')
plt.semilogy(x2,Z,'go',label= 'least square')
plt.legend()
plt.grid()
plt.show()

plt.figure(6)
plt.title('Function: cos(cos(x)) coeffs vs n loglog')
plt.xlabel(r'$n$')
plt.ylabel(r'$\ cos(cos(x) \ Coeffs$')
plt.loglog(x2,F2,'ro',label = 'Fourier series')
plt.loglog(x2,Z,'go',label = 'least square')
plt.legend()
plt.grid()
plt.show()

print("The max deviation in exp(x) is " + str(max(F1 - Y)))
print("The max deviation in cos(cos(x) is " + str(max(F2 - Z)))

# plotting the original and computed graph of cos(cos(x))
plt.figure(1) 
plt.title('Function : cos(cos(x)')
plt.scatter(x,np.dot(a,Z),color = 'green')
plt.grid(True)
plt.xlabel(r'$x$')
plt.ylabel(r'$cos(cos(x))$')
plt.plot(x0,y0,label = 'Original graph')
plt.plot(x,np.dot(a,Z),'g',label = 'Computed graph')
plt.legend(loc = 'upper right')
plt.show()

# plotting the original and computed graph of e(x)
plt.figure(2)
plt.title('Function : exp(x)') 
plt.grid(True)
plt.xlabel(r'$x$')
plt.ylabel(r'$e(x)$')
plt.semilogy(x0,y1,label = 'Original graph')
plt.semilogy(x,np.dot(a,Y),'g',label = 'Computed graph')
plt.legend(loc = 'upper right')
plt.show()





