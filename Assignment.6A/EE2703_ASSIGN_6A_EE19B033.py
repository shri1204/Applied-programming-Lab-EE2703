''' _________________________________

         ASSIGNMENT-6 [EE2703]
              EE19B033 
           KORRA SRIKANTH
      ____________________________
'''

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if (len(sys.argv)== 7) :
        n = int(sys.argv[1])         # Length of the tube-light.
        M = int(sys.argv[2])         # Number of Electrons injecter per turn.
        nk= int(sys.argv[3])         # Number of turns of simulation.
        uo= int(sys.argv[4])         # Threshold Velocity of Electrons.
        p = float(sys.argv[5])       # Probability of Ionization.
        Msig = float(sys.argv[6])    # Std-deviation of 'M'.
        print("Using user provided Arguments!!")
else :      # Default Values :
        n = 100                      
        M = 5                       
        nk= 500                    
        uo= 7                       
        p = 0.5                     
        Msig = 0.2  
        print("Using Default Arguments!!")
                 
# Initializing the vectors to zeros:
xx = np.zeros(n*M)                   
u  = np.zeros(n*M)
dx = np.zeros(n*M)

# Creating Three empty lists:
X=[]                                
V=[]
I=[]

for k in range(1,nk) :
                 ii = np.where(xx>0)[0]     # indices where electrons are present.                  
                 dx[ii] = u[ii] + 0.5       # updating the displacement.                 
                 xx[ii] = xx[ii] + dx[ii]   # updating the position.                 
                 u[ii]  = u[ii] + 1         # updating the velocity.
                 
                 # indices of electrons which reached the anode:
                 aa = np.where(xx > n)[0]                     
                 xx[aa] = 0        # setting the position to be 0.                           
                 u[aa] = 0         # setting the velocity to be 0.                          
                 dx[aa] = 0        # setting the displacement to be 0.                           

                 # indices of electrons whose velocity is greater than threshold:
                 kk = np.where(u >= uo)[0]                    
                 ll=np.where(np.random.rand(len(kk))<=p)[0] 
                 # kl contains the indices of electrons which undergo collisions:

                 kl=kk[ll] 
                 # Setting the velocity of collided electrons to 0:                 
                 u[kl] = 0                                    

                 # setting position of collided electrons: 
                 xx[kl] = xx[kl] - dx[kl]*np.random.rand()
                 # Updating the Intensity list:
                 I.extend(xx[kl].tolist())   
                 
                 # Injected electrons:   
                 m = int(np.random.randn()*Msig + M)
                 # Empty indices in the light:
                 ee = np.where(xx == 0)[0]                    
                 
                 if len(ee) >= m:                             
                       start = np.random.randint(len(ee))     # Choosing a random starting index.
                       xx[ee[start: m+start]] = 1             # Filling elecrons in the range of indices.
                       u[ee[start-m:start]] = 0               # setting their velocity to be zero.
                 else :
                       xx[ee] = 1                            
                       u[ee]  = 0
                 filled = np.where(xx > 0)
                 # Updating the Position list:
                 X.extend(xx[filled].tolist()) 
                 # Updating the Velocity list:               
                 V.extend(u[filled].tolist())               

                      

# Electron Density plot:
plt.figure(0)                                              
c,bins,xpos = plt.hist(X,bins=n)
plt.title('Electron Density')
plt.show()

# Light Intensity Plot:
plt.figure(1)                                               
plt.hist(I,bins=n)
plt.title('Light Intensity')
plt.show()

# Tabulating Emission Counts:
xpos = 0.5*(bins[0:-1]+bins[1:])
d = {'Position' :xpos, 'count':c }  
p= pd.DataFrame(data=d)
pd.set_option("display.max_rows",None,"display.max_columns",None)
print(p)



# Electron-Phase plot:
plt.figure(2)    
plt.plot(xx,u,'ro-')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Electron-Phase space')
plt.show()


