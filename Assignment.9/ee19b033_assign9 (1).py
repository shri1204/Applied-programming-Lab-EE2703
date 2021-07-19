'''__________________________________

       ASSIGNMENT-9 [EE2703]
          EE19B033
        KORRA SRIKANTH
    __________________________
'''
# Imports-
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

# Global Variable
PI = np.pi

def plotSpectrum(figTitle, w, Y, magStyle='bo-', phaseStyle='ro', xLimit=None, yLimit=None):
    plt.suptitle(figTitle)
    plt.subplot(2,1,1)
    plt.plot(w, abs(Y), magStyle, lw=2)
    plt.ylabel(r"$\|Y\|$")
    plt.grid()
    plt.xlim(xLimit)
    plt.ylim(yLimit)
    plt.subplot(2,1,2)
    ii=np.where(abs(Y)>1e-3)
    plt.plot(w[ii],np.angle(Y[ii]),'ro',lw=2)
    plt.xlim(xLimit)
    plt.ylabel(r"$\angle Y$")
    plt.xlabel(r"$\omega\ \to$")
    plt.grid()
    plt.show()


   #####QUESTION:1 EXAMPLE  - sin(sqrt(2)t)######
 

t1=np.linspace(-PI,PI,65);t1=t1[:-1]   # The interval used to find the DFT
t2=np.linspace(-3*PI,-PI,65);t2=t2[:-1]
t3=np.linspace(PI,3*PI,65);t3=t3[:-1]
y = np.sin(np.sqrt(2)*t1)
plt.figure(figsize=(12,6))
plt.plot(t2,np.sin(np.sqrt(2)*t2),'r',lw=3)
plt.plot(t1,np.sin(np.sqrt(2)*t1),'b',lw=3)   # The interval t1 is plotted in a different colour.
plt.plot(t3,np.sin(np.sqrt(2)*t3),'r',lw=3)
plt.ylabel(r"$y$",size=16)
plt.xlabel(r"$t$",size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)$")
plt.grid(True)
plt.show()


# plotting the function on interval t1 is repeated periodically
t1=np.linspace(-PI,PI,65);t1=t1[:-1]    # The interval used to find the DFT.
t2=np.linspace(-3*PI,-PI,65);t2=t2[:-1]
t3=np.linspace(PI,3*PI,65);t3=t3[:-1]
y = np.sin(np.sqrt(2)*t1)        # The function on interval t1 is repeated periodically.
plt.figure(figsize=(12,6))
plt.plot(t2,y,'ro',lw=2)
plt.plot(t1,y,'bo',lw=2)
plt.plot(t3,y,'ro',lw=2)
plt.legend(('Repeated wave','Part of the wave that is taken for finding DFT'))
plt.ylabel(r"$y$",size=16)
plt.xlabel(r"$t$",size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)$ sampled and repeated")
plt.grid(True)
plt.show()


# plotting the function on interval t1 is windowed and repeated periodically
t1=np.linspace(-PI,PI,65);t1=t1[:-1]   # The interval used to find the DFT.
t2=np.linspace(-3*PI,-PI,65);t2=t2[:-1]
t3=np.linspace(PI,3*PI,65);t3=t3[:-1]
n=np.arange(64)
wnd=fft.fftshift(0.54+0.46*np.cos(2*PI*n/63))
y=np.sin(np.sqrt(2)*t1)*wnd       # The function on interval t1 is windowed and repeated periodically
plt.figure(figsize=(12,6))
plt.plot(t2,y,'ro',lw=2)
plt.plot(t1,y,'bo',lw=2)
plt.plot(t3,y,'ro',lw=2)
plt.legend(('Repeated wave','Part of the windowed wave that is taken for finding DFT'))
plt.ylabel(r"$y$",size=16)
plt.xlabel(r"$t$",size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)$ windowed, sampled and repeated")
plt.grid(True)
plt.show()


# plotting the spectrum Without windowing
t = np.linspace(-PI, PI, 65)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = np.sin(np.sqrt(2)*t1)
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/64.0
w = np.linspace(-PI*fmax, PI*fmax, 65)[:-1]
plotSpectrum(r"Spectrum of $sin(\sqrt{2}t)$", w, Y, xLimit=[-10, 10])

#plotting the Windowing with Hamming Window
t = np.linspace(-PI, PI, 65)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = np.arange(64)
wnd=fft.fftshift(0.54+0.46*np.cos(2*PI*n/63))
y = np.sin(np.sqrt(2)*t) *wnd
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/64.0
w = np.linspace(-PI*fmax, PI*fmax, 65)[:-1]
plotSpectrum(r"Spectrum of $sin(\sqrt{2}t) * w(t)$", w, Y, xLimit=[-8, 8])


     ##### QUESTION 2 - SPECTRUM OF (cos(0.86 t))**3#####

 
#plotting the spectrum Without windowing
t = np.linspace(-4*PI, 4*PI, 257)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = np.cos(0.86*t)**3
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/256.0
w = np.linspace(-PI*fmax, PI*fmax, 257)[:-1]
plotSpectrum(r"Spectrum of $cos^3(0.86t)$", w, Y, xLimit=[-4, 4])

#plotting the spectrum Windowing with Hamming Window
t = np.linspace(-4*PI, 4*PI, 257)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = np.arange(256)
wnd=fft.fftshift(0.54+0.46*np.cos(2*PI*n/255))
y = (np.cos(0.86*t))**3 * wnd
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/256.0
w = np.linspace(-PI*fmax, PI*fmax, 257)[:-1]
plotSpectrum(r"Spectrum of $cos^3(0.86t) * w(t)$", w, Y, xLimit=[-4, 4])


      ##### ESTIMATION OF OMEGA,DELTA IN cos(wt + d)######


def estimateWandD(w, wo, Y, do, pow=2):
    wEstimate = np.sum(abs(Y)**pow * abs(w))/np.sum(abs(Y)**pow) # weighted average
    print("wo = {:.03f}\t\two (Estimated) = {:.03f}".format(wo, wEstimate))
    t = np.linspace(-PI, PI, 129)[:-1]
    y = np.cos(wo*t + do)
    c1 = np.cos(wEstimate*t)
    c2 = np.sin(wEstimate*t)
    A = np.c_[c1, c2]
    vals = lstsq(A, y)[0]
    dEstimate = np.arctan2(-vals[1], vals[0])
    print("delta = {:.03f}\t\tdelta (Estimated) = {:.03f}".format(do, dEstimate))


# Question 3 - Estimation of w, d in cos(wt + d)
wo = 1.35
d = PI/2
t = np.linspace(-PI, PI, 129)[:-1]
trueCos = np.cos(wo*t + d)
fmax = 1.0/(t[1]-t[0])
n = np.arange(128)
wnd=fft.fftshift(0.54+0.46*np.cos(2*PI*n/127))
y = trueCos.copy()*wnd
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/128.0
w = np.linspace(-PI*fmax, PI*fmax, 129)[:-1]
plotSpectrum(r"Spectrum of $cos(\omega_o t + \delta) \cdot w(t)$", w, Y, xLimit=[-4, 4])
estimateWandD(w, wo, Y, d, pow=1.75)


# Question 4 - Estimation of w, d in noisy cos(wt + d)
trueCos = np.cos(wo*t + d)
noise = 0.1*np.random.randn(128)
n = np.arange(128)
wnd=fft.fftshift(0.54+0.46*np.cos(2*PI*n/127))
y = (trueCos + noise)*wnd
fmax = 1.0/(t[1]-t[0])
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/128.0
w = np.linspace(-PI*fmax, PI*fmax, 129)[:-1]
plotSpectrum(r"Spectrum of $(cos(\omega_o t + \delta) + noise) \cdot w(t)$", w, Y, xLimit=[-4, 4])
estimateWandD(w, wo, Y, d, pow=2.5)


###### QUESTION 5 - DFT OF CHIRP ######


# chirp function used
def chirp(t):
    return np.cos(16*(1.5*t + (t**2)/(2*PI)))

#plotting the function without windowimg
N = 1024
t = np.linspace(-PI, PI, N+1);t=t[:-1]
x = chirp(t)
t = np.linspace(-np.pi,np.pi,N+1);t = t[:-1]
t1 = np.linspace(-3*np.pi,-np.pi,N+1);t1 = t1[:-1]
t2 = np.linspace(np.pi,3*np.pi,N+1);t2 = t2[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = (np.cos(16*t*(1.5+t/(2*np.pi))))
plt.figure(figsize=(12,6))
plt.grid()
plt.title(r'$cos(16t(1.5+\frac{t}{2\pi}))w(t)$')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.plot(t,y,'r')
plt.plot(t1,y,'b')
plt.plot(t2,y,'b')
plt.show()

#plotting the function without windowimg
N = 1024
t = np.linspace(-PI, PI, N+1);t=t[:-1]
x = chirp(t)
t = np.linspace(-np.pi,np.pi,N+1);t = t[:-1]
t1 = np.linspace(-3*np.pi,-np.pi,N+1);t1 = t1[:-1]
t2 = np.linspace(np.pi,3*np.pi,N+1);t2 = t2[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = (np.cos(16*t*(1.5+t/(2*np.pi))))
n = np.arange(N)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
y = y*wnd
plt.figure(figsize=(12,6))
plt.grid()
plt.title(r'$cos(16t(1.5+\frac{t}{2\pi}))w(t)$')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.plot(t,y,'r')
plt.plot(t1,y,'b')
plt.plot(t2,y,'b')
plt.show()


#plotting the spectrum Without windowing
fmax = 1.0/(t[1]-t[0])
X = fft.fftshift(fft.fft(x))/1024.0
w = np.linspace(-PI*fmax, PI*fmax, 1025);w=w[:-1]
plotSpectrum(r"DFT of $cos(16(1.5 + \frac{t}{2\pi})t)$", w, X, 'bo-', 'ro', [-100,100])

#plotting the spectrum Windowing with Hamming Window
n = np.arange(1024)
wnd=fft.fftshift(0.54+0.46*np.cos(2*PI*n/1023))
x = chirp(t)*wnd
X = fft.fftshift(fft.fft(x))/1024.0
plotSpectrum(r"DFT of $cos(16(1.5 + \frac{t}{2\pi})t) \cdot w(t)$", w, X, 'bo-', 'ro', [-100, 100])


    ####### TIME FREQUENCY PLOT #######


#plotting the surface-plot Without windowing 
t = np.linspace(-PI,PI,1025)
t = t[:-1]
dt = t[1]-t[0]
fmax = 1/dt
t = np.array(np.split(t, 16))    # The entire 1024 elements are split into 16 disjoint sets of 64 elements each.
n = np.arange(64)
wnd=fft.fftshift(0.54+0.46*np.cos(2*PI*n/63))
y = np.cos(16*t*(1.5 + (t/(2*PI))))
y[0]=0
y=fft.fftshift(y)
Y=fft.fftshift(fft.fft(y))/64.0
w=np.linspace(-PI*fmax,PI*fmax,65);w=w[:-1]

n = np.arange(0,1024,64)
fig1 = plt.figure(4)
ax = p3.Axes3D(fig1)
plt.title('Frequency vs Time surface plot')
ax.set_xlabel('Frequency ($\omega$)')
ax.set_ylabel('Time Block')
ax.set_xlim([-100,100])
ax.set_zlabel('DFT of signal')
x,y = np.meshgrid(w,n)
x[x>100]= np.nan      # Without this and the next line, the surface plot overflows due to the setting of xlim.
x[x<-100]= np.nan
surf = ax.plot_surface(x, y, abs(Y), rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0, antialiased=False)
plt.show()


#plotting the surface-plot of Windowed function
t = np.linspace(-PI,PI,1025)
t = t[:-1]
dt = t[1]-t[0]
fmax = 1/dt
t = np.array(np.split(t, 16))    # The entire 1024 elements are split into 16 disjoint sets of 64 elements each.
n = np.arange(64)
wnd=fft.fftshift(0.54+0.46*np.cos(2*PI*n/63))
y = np.cos(16*t*(1.5 + (t/(2*PI))))
y = y * wnd
y[0]=0
y=fft.fftshift(y)
Y=fft.fftshift(fft.fft(y))/64.0
w=np.linspace(-PI*fmax,PI*fmax,65);w=w[:-1]

n = np.arange(0,1024,64)
fig1 = plt.figure(4)
ax = p3.Axes3D(fig1)
plt.title('Frequency vs Time surface plot')
ax.set_xlabel('Frequency ($\omega$)')
ax.set_ylabel('Time Block')
ax.set_xlim([-100,100])
ax.set_zlabel('DFT of signal')
x,y = np.meshgrid(w,n)
x[x>100]= np.nan      # Without this and the next line, the surface plot overflows due to the setting of xlim.
x[x<-100]= np.nan
surf = ax.plot_surface(x, y, abs(Y), rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0, antialiased=False)
plt.show()
