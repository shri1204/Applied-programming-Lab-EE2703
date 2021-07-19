
''' ____________________________
   
       ASSIGNMENT 8 [EE2703]
          EE19B033
        KORRA SRIKANTH
      ___________________
'''
      
# Importing necessary libraries
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

# Global Variable
PI = np.pi

# Plotting Spectrum
def plots(w,Y,xlim,title,xlabel,ylabel1,ylabel2,presentfreq):
    plt.figure()
    plt.subplot(2,1,1)                            
    plt.plot(w,abs(Y))
    plt.title(title)                                      
    plt.ylabel(ylabel1)                         
    plt.xlim([-xlim,xlim])                        
    plt.grid()   
                                       
    plt.subplot(2,1,2)                            
    PM=np.where(abs(Y)>presentfreq)                    
    plt.plot(w[PM],np.angle(Y[PM]),'go')
    plt.xlim([-xlim,xlim])                        
    plt.xlabel(xlabel)                           
    plt.ylabel(ylabel2)                          
    plt.grid()                                  
    plt.show() 
        
                             
# Spectrum of sin(5t)
x = np.linspace(0, 2*PI, 128);x = x[:-1]
w = np.linspace(-64, 63, 128);w = w[:-1]
f1=np.sin(5*x)
F1= fft.fftshift(fft.fft(f1))/128.0
plots(w,F1,15,'Spectrum of sin(5t)',"k","|Y|","Phase of Y",1e-3)


# AM Modulation with (1 + 0.1cos(t))cos(10t)
x = np.linspace(-4*PI, 4*PI, 513);x = x[:-1]
w = np.linspace(-64, 64, 513);w = w[:-1]
f2 = (1+0.1*np.cos(x))*np.cos(10*x)
F2 = fft.fftshift(fft.fft(f2))/512.0
plots(w,F2,15,"AM Modulation with $(1+0.1cos(t))cos(10t)","k","|Y|","Phase of Y",1e-3)


# Spectrum of sin^3(t)
x = np.linspace(-4*PI, 4*PI, 513);x = x[:-1]
w = np.linspace(-64, 64, 513);w = w[:-1]
f3 = (np.sin(x))**3
F3 = fft.fftshift(fft.fft(f3))/512.0
plots(w,F3,15,"Spectrum of sin^3(t)","k","|Y|","Phase of Y",1e-3)

# Spectrum of cos^3(t)
x = np.linspace(-4*PI, 4*PI, 513);x = x[:-1]
w = np.linspace(-64, 64, 513);w = w[:-1]
f4 = (np.cos(x))**3
F4 = fft.fftshift(fft.fft(f4))/512.0
plots(w,F4,15,"Spectrum of cos^3(t)","k","|Y|","Phase of Y",1e-3)


# Spectrum of cos(20t + 5cos(t))
x = np.linspace(-4*PI, 4*PI, 513);x = x[:-1]
w = np.linspace(-64, 64, 513);w = w[:-1]
f5 = np.cos(20*x + 5*np.cos(x))
F5 = fft.fftshift(fft.fft(f5))/512.0
plots(w,F5,40,"Spectrum of cos(20*t+5*cos(t))","k","|Y|","Phase of Y",1e-3)


# Spectrum of Gaussian
# Phase and Magnitude of estimated Gaussian Spectrum
t = np.linspace(-4*PI, 4*PI, 257);t = t[:-1]
w = np.linspace(-64, 64, 257);w = w[:-1]
f6 = np.exp(-(t**2)/2)
F6 = fft.fftshift(fft.fft(fft.ifftshift(f6)))*8/512.0
plots(w,F6,15,"Spectrum of Estimated Gauss function","k","|Y|","Phase of Y",1e-3)

t = np.linspace(-4*PI, 4*PI, 513);t = t[:-1]
w = np.linspace(-64, 64, 513);w = w[:-1]
f6 = np.exp(-(t**2)/2)
F6 = fft.fftshift(fft.fft(fft.ifftshift(f6)))*4/128.0
plots(w,F6,15,"Spectrum of Gauss function","k","|Y|","Phase of Y",1e-3)

#expected gaussians
t = np.linspace(-4*PI, 4*PI, 513);t = t[:-1]
w = np.linspace(-64, 64, 513);w = w[:-1]
trueY = np.exp(-(w**2)/2)/np.sqrt(2*PI)
plots(w,trueY,15,"Spectrum of Expected Gauss function","k","|Y|","Phase of Y",1e-3)

print ("The maximum error between the expected and calculated gaussians are ", max(np.absolute(trueY  - F6)))




