'''   _________________________________

          ASSIGNMENT 5 [EE2703]
               EE19B033
             KORRA SRIKANTH
      __________________________________
'''
   
from pylab import *
from matplotlib.pyplot import *
import mpl_toolkits.mplot3d.axes3d as p3
from sys import argv


if (len(argv) ==4):
    Nx = int(argv[1])         # No. of steps along the x direction
    Ny = int(argv[2])         # No. of steps along the y direction
    r = float(argv[3])
    print("Using user provided params")
else:
    Nx = 25
    Ny = 25  
    r = 8
    print("Using default parameters") 

# Initializing the potential matrix
phi = zeros((Nx,Ny))
Niter = 1500
x = linspace(-Nx/2,Nx/2,Nx)
y = linspace(Nx/2,-Nx/2,Ny)
Y,X = meshgrid(y,x)

# Finding out the points with 1 V potential
Z = where(X*X + Y*Y <= r*r )

# Marking potential = 1V points in the phi matrix 
for i in range(Z[0].size):
    phi[Z[0][i],Z[1][i]] = 1
  
# Plotting the 1V points and the contour plot of phi matrix
figure(1) 
title('1V Potential and the contour plot of potential matrix')
xlabel(r'$x$')
ylabel(r'$y$')
plot(x[Z[0]],y[Z[1]],'ro', label = '1V Potential')
contour(X,Y,phi)
legend()
grid()
show()

# iteration which calculates the steady state potential array
error=[]
e1 = zeros((30))
# iteration which calculates the steady state potential array
for  i in range(Niter) :
    oldphi = phi.copy()
    phi[1:-1,1:-1] = 0.25*(phi[1:-1,0:-2]+phi[1:-1,2:]+phi[0:-2,1:-1]+phi[2:,1:-1])
    phi[1:-1,0]=phi[1:-1,1]
    phi[1:-1,Nx-1]=phi[1:-1,Nx-2]
    phi[0,1:-1]=phi[1,1:-1]
    phi[0,0]=phi[0,1]
    phi[0,Nx-1]=phi[0,Nx-2]
    phi[Ny-1,1:-1]=0.0
    phi[Z]=1.0
    error.append((abs(oldphi-phi)).max())


for i in range(0,30):
    e1[i] = error[i*50] # Error for every 50th point

# Plotting the semilogy plot of error along with every 50th point
R = list(range(Niter))
figure(2) 
xlabel(r'$Iterations$')
ylabel(r'$Error$')
semilogy(R,error,label = 'error')
semilogy(linspace(1,Niter-30,30),e1,'ro')
title(r'Semilogy of error along with every 50th point')
grid()
legend()
show()

# Plotting the loglog plot of error  
figure(3)
xlabel(r'$Iterations$')
ylabel(r'$Error$')
loglog(R,error,label = 'error')
title(r'Error in loglog scale for all iteration ')
grid()
legend()
show()


expe = error[500:]
logy = log(expe)
n = linspace(501,Niter,Niter-500)
L1 = ones((Niter-500,2))
L1[:,0]= n

expe1 = error[:]
logy1 = log(expe1)
n1 = linspace(1,Niter,Niter)
R1 = ones((Niter,2))
R1[:,0]= n1

# Fitting the error to an exponential for > 500 iterations
err = linalg.lstsq(L1,logy,rcond=-1)[0] 
error_fit =  err[1] + err[0]*n


# Fitting the error to an exponential
err1= linalg.lstsq(R1,logy1,rcond=-1)[0] 
error_fit1 =  err1[1] + err1[0]*n1


# Plotting the error fits
figure(4)
title('Error in semilog scalefor all iterations')
xlabel(r'$Iterations$')
ylabel(r'$Error$')
plot(linspace(1,Niter,Niter),log(error),"g",label = 'errors')
plot(n,error_fit,"ro",label = 'fit error > 500 iter')
plot(n1,error_fit1,"b--",label = 'fit error')
grid()
legend()
show()

# Plotting the 3D surface plot of potential
fig4 = figure(5) 
ax = p3.Axes3D(fig4)
title('The 3-D Surface Plot of the potential')
ylabel(r'$GROUND$')
surf = ax.plot_surface(X,Y,phi,rstride=1,cstride=1,cmap = cm.jet,label = 'Potential')
show()



# Contour plot of potential
figure(6)
title('Contour plot of the potential') 
xlabel(r'$x$')
ylabel(r'$y$')
plot(x[Z[0]],y[Z[1]],'ro', label = '1V Potential')
clabel(contour(x,y,phi) , inline = True , fontsize = 8)
grid()
legend()
show()

# Initializing the current vectors
Jx = zeros((Nx,Ny)) 
Jy = zeros((Nx,Ny))
# Calculating the current
Jy[1:-1,1:-1] = 0.5*( phi[1:-1,0:-2] - phi[1:-1,2:])
Jx[1:-1,1:-1] = 0.5*( phi[0:-2,1:-1] - phi[2:,1:-1])

# Plotting the current vectors using quiver
figure(7) 
title('Quiver plot of the current densities')
xlabel(r'$x$')
ylabel(r'$y$')
plot(x[Z[0]],y[Z[1]],'ro', label = '1V potential')
quiver(y,x,-Jy[::-1,:],-Jx[::-1,:],scale=1.8,scale_units='inches',label="Current density")
grid()
legend()
show()


T = zeros((Nx,Ny))
T[:,:] = 300
sigma = 6*(10**7)
kappa = 385
for i in range(Niter):
    T[1:-1,1:-1] = 0.25*(T[1:-1,0:-2] + T[1:-1,2:] + T[0:-2,1:-1] + T[2:,1:-1] + (((Jx**2)[1:-1,1:-1] + (Jy**2)[1:-1,1:-1])*sigma*(16*(10**-8)))/kappa)
    T[1:-1,0]=T[1:-1,1]
    T[1:-1,Nx-1]=T[1:-1,Nx-2]
    T[0,1:-1]=T[1,1:-1]
    T[Z] = 300.0

fig1=figure(8)
ax=p3.Axes3D(fig1)
title('The 3-D surface plot of the temperature')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Temperature')
ax.plot_surface(X, Y, T, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
show()


J_sq = Jx**2 + Jy**2

figure(9)
title('Contour plot of the heat generated')
cp = contour(-Y,-X,J_sq)
clabel(cp,inline=True,colors='r')
xlabel('x')
ylabel('y')
grid()
show()


