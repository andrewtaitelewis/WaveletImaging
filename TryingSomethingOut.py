#Importing Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import os 
import PIL
#=====================================
#One Dimensional Wavelet
#=====================================
#Mexican hat wavelet
def mexicanHat1D(t,sigma,center):
    '''
    1D mexican hat wavelet with respect to t and sigma
    '''
    
    t = np.asarray(t)
    return (2/(np.sqrt(3*sigma)*np.pi**(1/4.0)))*(1 - ((t-center)/sigma)**2)*np.exp((-(t-center)**2)/(2*sigma**2))
#Random signal generator
def randomSignal1D(length,start):
    '''
    Returns a random signal
    Params:
    length: length of the array that will be returned
    start: the y value at the start
    '''
    xs = []
    returnedArray = []
    
    for i in range(length):
       
        if np.random.rand() < 0.5:
            start = start - np.random.rand()
        else:
            start = start + np.random.rand()
        returnedArray.append(abs(start))
        xs.append(i)
    return xs,returnedArray
#Deconstructs 
def waveletDeconstructor(scales,xs,data):
    '''
    Deconstructs a data stream using the scales array and the mexican hat wavelet
    Params:
    -------
    Scales, sigmas used in the wavelet
    xs: x values that the function has been evaluated at
    data: raw data stream, same length of xs
    Returns:
    --------
    yValues: a y value of the reconstructed function
    '''
    yValues = np.zeros(len(data))
    for scale in scales:
        
        coefficients = []
        #Breaks up the data
        for i in xs:
            coefficients.append(np.dot(data,mexicanHat1D(xs,scale,i)))
        #reconstructs the data
        for i,j in zip(xs,coefficients):
            yValues += mexicanHat1D(xs,scale,i)*j


    return yValues/sum(abs(yValues))
#Deconstructing with different centers
def waveletDeconstructor2(scales,xs,data,centers):
    '''
    Deconstructs a data stream using the scales array and the mexican hat wavelet
    Params:
    -------
    Scales, sigmas used in the wavelet
    xs: x values that the function has been evaluated at
    data: raw data stream, same length of xs
    Returns:
    --------
    yValues: a y value of the reconstructed function
    '''
    yValues = np.zeros(len(data))
    for scale in scales:
        
        coefficients = []
        #Breaks up the data
        for i in centers:
            coefficients.append(np.dot(data,mexicanHat1D(xs,scale,i)))
        #reconstructs the data
        for i,j in zip(centers,coefficients):
            yValues += mexicanHat1D(xs,scale,i)*j

    return yValues/sum(abs(yValues))

def mexicanHat2D(x,y,cent,sigma):
    ''' 
    Two dimensional mexican hat wavelet
    Params:
    -------
    x: x coordinate
    y: y coordinate
    cent: a pair with the x,y coordiates of where the wavelet is centered
    sigma: sigma
    Returns:
    --------
    zValue: the value of the wavelet
    '''
    #Getting the proper x and y 
    x = (x - cent[0])/2
    y = y - cent[1]
    
    #I'm going to split up the calculation
    result = 1/np.pi
    result = result/(sigma**2)
    intermitant = (1-(x**2 +y**2)/(sigma**2)*(1/2))
    result = result*intermitant
    intermitant  = ((x**2) + (y**2))/(2*sigma**2)*(-1)
    result = result*np.exp(intermitant)
    return result

    
#The Code
#====================================
np.random.seed(0)       #Just to make sure I get the same data again and again
t,y = randomSignal1D(50,0)      #The sample data
y = np.asarray(y)

y = y/sum(abs(y))        #Normalizing the data
plt.plot(t,y)   #Plotting the raw data
plt.show()

#Now a wavelet deconstuction
reconstructedData = waveletDeconstructor([1,2,3],t,y)

plt.plot(t,reconstructedData, label = 'Reconstructed Data')
plt.plot(t,y, label = ' Original Data')
plt.legend()
plt.show()

#Now a few more scales
everyOther = np.linspace(2,47,120)
reconstructedData = waveletDeconstructor2(np.linspace(.01,1,5),t,y,everyOther)
plt.plot(t,reconstructedData, label = 'Reconstructed Data')
plt.plot(t,y, label = ' Original Data')
plt.legend()
plt.show()

#What does my wavelet look like
xs = np.linspace(-5,5,41)
ys = mexicanHat1D(xs,1,0)
plt.axhline(0)
plt.plot(xs,ys)
plt.show()

#=====================================
#Two Dimensional Wavelet
#=====================================
#Two dimensional mexican hat wavelet

    
#Code
#======================================
#Testing the mexican hat
xs = np.linspace(-5,5,21)
ys = np.linspace(-5,5,21)

zValues = []
for i in xs:
    appendList = []
    for j in ys:
        appendList.append(mexicanHat2D(i,j,(0,0),1))
    zValues.append(appendList)

#Now plot
zValues = np.asarray(zValues)
zValues = zValues/sum(sum(zValues))         #Normalize 
plt.contourf(xs,ys,zValues)
plt.colorbar()
plt.show()


'''
#Our centroid tracker
x = np.linspace(xBounds[0],xBounds[1],gridSize[0])
y = np.linspace(yBounds[0],yBounds[1],gridSize[1])
xTrack, yTrack = centroidFitter(1,zs,x,y)
trackedCoords = (zip(xTrack,yTrack))

x,y = np.meshgrid(x,y)



plt.show()
#convert to numpy array 

xSteps,ySteps = np.asarray(xSteps),np.asarray(ySteps)
xTrack,yTrack = np.asarray(xTrack), np.asarray(yTrack)
plt.plot(xSteps,xSteps-xTrack, 'o', label = 'x Residuals')
plt.plot(ySteps,ySteps-yTrack, 'o', label = 'y Residuals')
plt.legend()
plt.title('Residuals of Centroid Method')
plt.axhline(0)

plt.show()
CentroidResiduals = np.sqrt((xSteps-xTrack)**2 + (ySteps-yTrack)**2)

#Now time for our non linear least squares fit

#2d gaussian function to determine coefficients
def gauss2dFlat(X,xCenter,yCenter,sigma):
    x,y = X
    
    xcomp = ((x-xCenter)**2)/(2*sigma**2) 
    ycomp = ((y-yCenter)**2)/(2*sigma**2)
    result = np.exp(-(xcomp+ycomp))
  
    return result.flatten()

firstFrame = (zs[15].flatten())

#now fit it, our guess
popt, cov = optimize.curve_fit(gauss2dFlat,(x,y),firstFrame)
X = (x,y)
guess = gauss2d(X,popt[0],popt[1],popt[2])


xTrack, yTrack = [],[]
#Guessing the residuals
for i in zs:
    if len(xTrack) == 0:
        prevX,prevY = 0,0
    else:
        prevX,prevY = xTrack[-1],yTrack[-1]
    popt, cov = optimize.curve_fit(gauss2dFlat,(x,y),i.flatten(),[prevX,prevY,1])
    xTrack.append(popt[0])
    yTrack.append(popt[1])

trackedCoords = (zip(xTrack,yTrack))


plt.ion()
for i,j in zip(zs,trackedCoords):
    xcoord,ycoord = j
    plt.clf()
    plt.contourf(x,y,i)
    plt.colorbar()
    plt.plot(xcoord,ycoord,'o',color = 'orange')
    plt.draw()
    plt.pause(0.01)

#Residual
plt.ioff()
plt.show()
#Comparing the 
xSteps,ySteps = np.asarray(xSteps),np.asarray(ySteps)
xTrack,yTrack = np.asarray(xTrack), np.asarray(yTrack)
NonLinearResiduals = np.sqrt((xSteps-xTrack)**2 + (ySteps-yTrack)**2)
plt.legend()
plt.plot(range(len(xSteps)),CentroidResiduals, '.', label = 'Centroid Residuals')
plt.plot(range(len(xSteps)),NonLinearResiduals, '.', label = "Non Linear Residuals")
plt.legend()
plt.title('Residuals of Centroid Method')
plt.axhline(0)
plt.show()'''