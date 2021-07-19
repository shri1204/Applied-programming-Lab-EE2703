''' ________________________________
     
        ASSIGNMENT-3 [EE2703]
             EE19B033
          KORRA SRIKANTH
      ___________________________
'''
# Importing Necessary libraries
import numpy as np
from pylab import *
import scipy.special as sp
from scipy.linalg import lstsq
import sys

#QUESTION: 1
#Run the following command in the terminal
#python generate_data.py

#QUESTION: 2
try:
	data = np.loadtxt("fitting.dat")       #put try else here
except IOError:
        sys.exit("fitting.dat not found! Please run the code in generate_data.py before you run this code.")
    
sigma=np.logspace(-1,-3,9)
sigma=around(sigma,3)

N,k = data.shape
t = np.linspace(0,10,N)

def g(t,A,B):
    return A*sp.jn(2,t) + B*t

f_true = g(t,1.05,-0.105)

#QUESTION: 3
for i in range(k-1):
   plot(t,data[:,i+1],label = '$\sigma$' +"="+ str(sigma[i]))
plot(t,f_true,label = r"True value")
xlabel(r'$t$',size=20)
ylabel(r'$f(t)+n$',size=20)
title(r'Q4:Fitted data with noise')
grid(True)
legend()
show()

#QUESTION: 4
plot(t,f_true,label = r"True value")
xlabel(r'$t$',size=15)
ylabel(r'$f(t)$',size=15)
title(r'Q4:Fitted Function To Given Data')
grid(True)
legend()
show()

#QUESTION: 5
errorbar(t[::5],data[::5,1],sigma[1],fmt='ro',label = r"Error bar")
xlabel(r"$t$",size=15)
ylabel(r'$f(t)+n$',size=15)
title(r"Q5:Data points for $\sigma$ = 0.1 along with exact function")
plt.legend()
plot(t,f_true,color="black",label = r"True value (f(t))")
legend()
grid()
show()

#QUESTION: 7
A = arange(0,2,0.1)
B = arange(-0.2,0,0.01)
fk =g(data[:,0],1.05,-0.105)

epsilon = np.zeros((len(A),len(B)))

for i in range(len(A)):
    for j in range((len(B))):
        epsilon[i,j] = np.mean(np.square(fk - g(t,A[i],B[j])))

#QUETION: 8
cp = plt.contour(A,B,epsilon,20)
plot(1.05,-0.105,"ro")
annotate(r"$Exact\ location$",xy=(1.05,-0.105))
xlabel(r"$A$",size=15)
ylabel(r"$B$",size=15)
title(r"Q8:Countour plot for $\epsilon_{ij}$")
grid(True)
show()     

#QUESTION: 6
M = np.zeros((N,2))
for i in range(N):
    M[i,0] = sp.jn(2,data[i,0])
    M[i,1] = data[i,0]

# QUESTION: 9
pred = []
Aerror=[]
Berror=[]

for i in range(k-1):
    pred,resid,rank,sig=lstsq(M,data[:,i+1])
    aerr = np.square(pred[0]-1.05)
    berr = np.square(pred[1]+0.105)   
    Aerror.append(aerr)
    Berror.append(berr)
#print(pred[0])
#print(pred[1])
#print(aerr)
#print(berr)
#print(Aerror)
#print(Berror)

#QUESTION: 10
plot(sigma,Aerror,"ro",linestyle="--", linewidth = 1,label=r"$Aerr$")
plt.legend()
plot(sigma,Berror,"co",linestyle="--",linewidth = 1,label=r"Berr")
xlabel(r"$\sigma_{noise}$",size=15)
ylabel(r"$MS\ Error$",size=15)
title("$Q10:Variation\ of\  error\  with\  noise$")
legend()
grid(True)
show()
 
#QUESTION: 11
loglog(sigma,Aerror,"r--")
errorbar(sigma,Aerror,np.std(Aerror),fmt="ro",label=r"$Aerr$")
loglog(sigma,Berror,"g--")
errorbar(sigma,Berror,np.std(Berror),fmt="go",label=r"$Berr$")
ylabel(r"$MS\ Error$",size=15)
title(r"$Q11:Variation\ of\ error\ with\ noise$")
xlabel(r"$\sigma_{noise}$",size=15)
legend()
grid(True)
show()
