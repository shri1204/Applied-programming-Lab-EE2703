'''
    ____________________________

        FINAL EXAM[EE2703]
          EE19B033
        KORRA SRIKANTH
 _____________________________________
 
'''
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

# Creating Meshgrid
x=np.linspace(0,2,3)
z=np.arange(0,1000,1)
X,Y,Z=np.meshgrid(x,x,z)            

# Declaring The Radius Of Wire And Number Of Sections
radius=10                  
N=100             
phi=np.linspace(0,2*np.pi,N)

# Position Vector
rijk=np.array((X,Y,Z))

# rl Vector
rl=np.vstack((radius*np.cos(phi).T,radius*np.sin(phi).T)).T
# dl Vector
dl=2*np.pi*radius/N*np.vstack((np.cos(phi).T,np.sin(phi).T)).T


'''
.T - Transposes The Array
np.vstack - Stacks Arrays In Sequence Vertically (Row Wise)
'''

m1=np.tile(rijk,(100,1,1,1)).reshape((100,3,3,3,1000)) 
m2=np.hstack((rl,np.zeros((100,1)))).reshape((100,3,1,1,1))


#5 Function To Calculate Rijkl
def calc(l):
    return np.linalg.norm(m1-m2,axis=1)

# Function To Calculate The Current Along x-direction And y-direction
def c_(x,y):
    return np.array([np.sin(phi),-np.cos(phi)])

# Currents Along x-direction And y-direction
i_x,i_y=c_(rl[:,0],rl[:,1])        

#3
# Plotting Current Vectors In x-y Plane
plt.figure(2)
plt.quiver(rl[:,0],rl[:,1],i_x,i_y,scale=40,headwidth=9,headlength=12,color='red')
plt.xlabel(r'x-axis')                        
plt.ylabel(r'y-axis')                  
plt.title('Current In Wire In x-y Plane')               
plt.grid()                                                           
plt.show()                                                           


# Calculating Rijkl Using 'calc' Function
R=calc(1)                                                   
cosA=np.cos(phi).reshape((100,1,1,1))                         
dl_x=dl[:,0].reshape((100,1,1,1))              # dl Vector Component Along x-direction
dl_y=dl[:,1].reshape((100,1,1,1))                       
Ax=np.sum(cosA*dl_x*np.exp(1j*R/10)*dl_x/R,axis=0)      
Ay=np.sum(cosA*dl_y*np.exp(1j*R/10)*dl_y/R,axis=0)     


#8
# Calculating Magnetic Field
Bz=(Ay[1,0,:]-Ax[0,1,:]-Ay[-1,0,:]+Ax[0,-1,:])/(4)            


#9
# Plotting Magnetic Field Along z-axis
plt.figure(3)
plt.loglog(z,np.abs(Bz))                                              
plt.xlabel(r'z-axis')                        
plt.ylabel(r'B(Magnetic Field)')             
plt.title('Magnetic Field Along z Axis')                  
plt.grid()                                                           
plt.show()                                                            


#10
# Solving For B Using Least Squares
A=np.hstack([np.ones(len(Bz[50:]))[:,np.newaxis],np.log(z[50:])[:,np.newaxis]])
log_c,b=np.linalg.lstsq(A,np.log(np.abs(Bz[50:])),rcond=None) [0]     
c=np.exp(log_c)                 
print("The Value Of b is:",b)
print("The Value Of c is:",c)


                      
