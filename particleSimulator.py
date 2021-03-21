'''
This code is going to be used to simulate a particle moving in or around a map, it can have constraints like boxes as well
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
#============================================
#Functions
#============================================
def gauss2d(indep,xCenter,yCenter,sigma):

    x,y = indep
    xcomp = ((x-xCenter)**2)/(2*sigma**2) 
    ycomp = ((y-yCenter)**2)/(2*sigma**2)
    
    return np.exp(-(xcomp+ycomp))
def gauss2dFlat(X,xCenter,yCenter,sigma):
    x,y = X
    
    xcomp = ((x-xCenter)**2)/(2*sigma**2) 
    ycomp = ((y-yCenter)**2)/(2*sigma**2)
    result = np.exp(-(xcomp+ycomp))
  
    return result.flatten()
def walkerWithBox(steps,radius,xs,ys,boxEscapeProb, startingCoords):
    '''
    A walker that may or may not get stuck in boxes
    Params:\\
    -------
    steps: int, number of steps in the chain \\
    radius: int radius of the circle to particle could jump to \\
    xs: list of all the xParitions that are sides of a box
    ys: list of all the YPartitions that are sides of a box
    boxEscapeProb: float, the probability of escaping from one box to another \\
    startingCoords: (float,float): where the walker starts \\
    Returns:\\
    --------
    list of [(x,y)] of steps
    '''
    #Unpacking the starting coords
    xCur,yCur = startingCoords
    xSteps = [xCur]
    ySteps = [yCur]
    #Initialize where we are
    ys = ys.tolist()
    xs = xs.tolist()
    #Temporary xs and ys
    tempx ,tempy = xs.copy(),ys.copy()
    tempx.append(xCur)
    tempy.append(yCur)
    tempx.sort()
    tempy.sort()
    #Get the indexes of where we are
    xIndex = tempx.index(xCur)
    yIndex = tempy.index(yCur)
    

    #End of initalization
    #=====================================
    
    #Algorithm,
    # 1. Try and move
    # 2. If move crosses a line, move with a probability
    # 2b. update bounds
    # 3 repeat until out of steps
    xSteps, ySteps = [xCur],[yCur]
    for i in range(steps):
        xStep,yStep = np.random.normal(0,radius,2)
        proposedXstep, proposedYstep = xCur + xStep, yCur + yStep
        #Put the propossed step in the list
        tempx = xs.copy()
        tempy = ys.copy()
        
        tempx.append(proposedXstep)
        tempy.append(proposedYstep)
        tempx.sort()
    
        tempy.sort()
        #if the indexes are both the same:
        if xIndex == tempx.index(proposedXstep) and yIndex == tempy.index(proposedYstep):
            xCur = proposedXstep
            yCur = proposedYstep
            xSteps.append(xCur)
            ySteps.append(yCur)
            continue
        #If the sorted index is different, we know it has crossed a boundary
        if xIndex != tempx.index(proposedXstep) or yIndex != tempy.index(proposedYstep):
            if np.random.rand() < boxEscapeProb:
                xCur = proposedXstep
                xIndex = tempx.index(proposedXstep)
                yCur = proposedYstep
                yIndex = tempy.index(proposedYstep)
                xSteps.append(xCur)
                ySteps.append(yCur)
                print(str(i) + "Jump!")
                
            else:
                
                continue
        #Append to list
        
        
        # Append to the chain
        #other





    return xSteps,ySteps
#Creates a grid of boxes 
def boxMapMaker(xBounds,yBounds,NumDivisions):
    '''
    Creates a map for the walker to explore based on the parameters\\
    Params:
    -------
    xBounds: tuple, (minX, maxX)
    yBounds: tuple, (minY, maxY)
    numDivisons: number of barriers in the grid for both x and y
    Returns:
    --------
    boxes: list x,y: a list of all the x paritions and y partitions
    '''
    #Unpacking the bounds
    xMin,xMax = xBounds
    yMin,yMax = yBounds
    xs = np.linspace(xMin,xMax,NumDivisions+1)
    ys = np.linspace(yMin,yMax,NumDivisions+1)
    return xs,ys
#Plots the journey of the particle
def interactivePlotter(xs,ys, xStep,yStep):
    '''
    A plotter that shows the journey of our simulated particle:
    Params:
    -------

    Returns:
    --------
    '''
    plt.ion()
    for i in range(len(xStep)):
    
        
        plt.plot(xStep[:i],yStep[:i],color = 'orange')
        for i,j in zip(xs,ys):
            plt.axhline(i)
            plt.axvline(j)
        if i == len(xStep) -1:
            plt.title("Done")
        plt.draw()
       
        plt.pause(0.0001)
        
        if i == len(xStep):
            
            continue
        plt.clf()
    
    
    plt.show()
    
        
   

    return
#Simulates data
def dataSimulator(xBounds,yBounds,grid,steps):
    '''
    Simulates the data using the walker and the grid bounds
    Paramters:
    ----------
    xbounds: tuple, (lowerx,upperx)
    ybounds: tuple, (lowery,uppery)
    grid: tuple, (xsize,ysize)
    steps: tuple     (xsteps,ysteps)
    Returns:
    --------
    arrayZs: A xsize by ysize array that contains the values of z (the height) at every given grid point

    '''
    #Unpacking the grid bounds
    lowerX,upperX = xBounds
    lowerY,upperY = yBounds
    xsize,ysize = grid
    xSteps,ySteps = steps
    #Making our np arrays
    x = np.linspace(lowerX,upperX,xsize)
    y = np.linspace(lowerY,upperY,ysize)
    x,y = np.meshgrid(x,y)
    
    arrayZs = []        #Array of heatmaps to return
    for i,j in zip(xSteps,ySteps):
        #Randomize some noise
        returnedZs = np.random.poisson(1,(xsize,ysize))
        returnedZs = returnedZs/np.amax(returnedZs)
        #Plot a gaussian
        a = (x,y)
        returnedZs += 3*gauss2d(a,i,j,1)
        arrayZs.append(returnedZs)
    return arrayZs
#Fits using a centroid
def centroidFitter(threshold,imageSeries,x,y):
    '''
    Guesses where a given 'cell' is using the centroid fitting method
    Parameters:
    -----------
    Threshold: float, if any of the z values are below this value, they will be set to zero
    imageSeries: array, an array of zs used to determine where the cell is in time
    x,y: the x y values of the grid, as it is discrete and not continuous
    Returns:
    --------
    XCoords,Ycoords: List,list two lists of the xCoordinates of the cell, and the y coordinates of the cell
    '''
    #We will try and threshold for the noise
    xCoords = []
    yCoords = []
    for i in imageSeries:
        #Treshold the images
        array = ((i > threshold)) * threshold
        #x centroid
        weight = 0
        xPixels = 0
        for j in array:
            
            weight += sum(j)
            xPixels += sum(x*j)
        xcoord = (xPixels/weight)
        #y centroid
        weight = 0
        yPixels = 0
        for j in np.transpose(array):
            weight += sum(j)
            yPixels += sum(y*j)
        ycoord = (yPixels/weight)
        #Append the new guess
        xCoords.append(xcoord)
        yCoords.append(ycoord)
    return xCoords,yCoords
#One dimensional Mexican hat Wavelet
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

#============================================
#Code
#============================================
'''
np.random.seed(0)       #Our Random Seed

xBounds = (-25,25)
yBounds = (-25,25)
gridSize = (100,100)
#Makes our box
xs,ys= (boxMapMaker(xBounds,yBounds,))
xSteps,ySteps = walkerWithBox(150,1.5,xs,ys,.05,(0,0))

interactivePlotter(xs,ys,xSteps,ySteps)

zs = (dataSimulator(xBounds,yBounds,gridSize,(xSteps,ySteps)))

#Plot counturf

x = np.linspace(xBounds[0],xBounds[1],gridSize[0])
y = np.linspace(yBounds[0],yBounds[1],gridSize[1])
x,y = np.meshgrid(x,y)
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
#Calculating the residuals
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

np.random.seed(0)
#1D Signal, Wavelet Transforms - Figure 1
#==========================================================
np.random.seed(0)   #Seeding the random number generator
t,y = randomSignal1D(50,0)      #The sample data
y = np.asarray(y)
y = y/sum(abs(y))        #Normalizing the data
plt.plot(t,y)   #Plotting the raw data
plt.show()

#Deconstucting the 1d Signal
#Now a wavelet deconstuction
reconstructedData = waveletDeconstructor([1,2,3],t,y)
plt.plot(t,reconstructedData, label = 'Reconstructed Data')
plt.plot(t,y, label = ' Original Data')
plt.legend()

#Plotting ====================================
fig1, (ax11,ax12) = plt.subplots(2)    #Wavelet Deconstruction Figure
#ax11 = raw data, reconstructed data
ax11.plot(t,reconstructedData, label = 'Reconstructed Data')
ax11.plot(t,y, label = 'Raw Data')
ax11.legend()
ax11.set_title('Raw 1D data and wavelet reconstruction')
#ax12 = residuals
ax12.plot(t,y-reconstructedData, '.')
ax12.axhline(0,color = 'black')
ax12.set_title('Residuals')
plt.show()

plt.clf()
#Simulation of Data - Figure 2
#===========================
#2D signal, no boxes
xBounds, yBounds = (-30,30),(-30,30)
gridSize = (100,100)
x = np.linspace(xBounds[0],xBounds[1],gridSize[0])
y = np.linspace(xBounds[0],xBounds[1],gridSize[0])
#No Grids
xPartition,yPartition = boxMapMaker(xBounds,yBounds,0)
xPartition,yPartition = np.asarray(xPartition),np.asarray(yPartition)
xTrack,yTrack = walkerWithBox(500,1,xPartition,yPartition,1,(0,0))
plt.plot(xTrack,yTrack)
plt.show()
plt.clf()
#Box with x,y = -2.5,2.5, 
xs,ys = np.asarray([-2.5,2.5]),np.asarray([-2.5,2.5])   #Box
xSteps,ySteps = walkerWithBox(500,1,xs,ys,0.1,(0,0))
plt.plot(xSteps,ySteps)
plt.axhline(-2.5)
plt.axhline(2.5)
plt.axvline(-2.5)
plt.axvline(2.5)
plt.show()
#Plot with noise,
zs = (dataSimulator(xBounds,yBounds,gridSize,(xSteps,ySteps)))
plt.contourf(zs[0])
plt.show()
fig2,(ax21,ax22,ax23) = plt.subplots(1,3)
fig2.figsize = (5*5,5)
ax21.plot(xTrack,yTrack)
ax21.set_aspect('equal')
ax21.set_xlim(-12,12)
ax21.set_ylim(-12,12)
ax22.plot(xSteps,ySteps)
ax22.set_xlim(-12,12)
ax22.set_ylim(-12,12)
ax22.axhline(-2.5)
ax22.axhline(2.5)
ax22.axvline(-2.5)
ax22.axvline(2.5)
ax22.set_aspect('equal')
ax23.contourf(zs[0])
ax23.set_aspect('equal')
plt.show()

#Centroid and Linear Least Squares Fiting - Figure 3
#===================================================
#Plot the Gaussian at the end of the track
plt.contourf(x,y,zs[-1])
plt.plot(xSteps[::15],ySteps[::15],'-.', label = 'Raw Data Path')

#Deconstruct it using centroid and linear least squares
x = np.linspace(xBounds[0],xBounds[1],gridSize[0])
y = np.linspace(yBounds[0],yBounds[1],gridSize[1])
xTrack, yTrack = centroidFitter(1,zs,x,y)
cenXTrack, cenYTrack = xTrack,yTrack
trackedCoords = (zip(xTrack,yTrack))
plt.plot(xTrack[::15],yTrack[::15],'-.', label = 'Centroid Fit Path')
#And Finally Least Squares
xSteps,ySteps,xTrack,yTrack = np.asarray(xSteps),np.asarray(ySteps),np.asarray(xTrack),np.asarray(yTrack)
CentroidResiduals = np.sqrt((xSteps-xTrack)**2 + (ySteps-yTrack)**2)
xTrack, yTrack = [],[]
#Calculating the residuals
x,y = np.meshgrid(x,y)
for i in zs:
    if len(xTrack) == 0:
        prevX,prevY = 0,0
    else:
        prevX,prevY = xTrack[-1],yTrack[-1]
    popt, cov = optimize.curve_fit(gauss2dFlat,(x,y),i.flatten(),[prevX,prevY,1])
    xTrack.append(popt[0])
    yTrack.append(popt[1])

trackedCoords = (zip(xTrack,yTrack))
plt.plot(xTrack[::15],yTrack[::15],'-.', label = 'Non Linear Least Squares Fit')
plt.show()

#And the residuals
steps = np.linspace(1,500,len(CentroidResiduals))
xSteps,ySteps,xTrack,yTrack = np.asarray(xSteps),np.asarray(ySteps),np.asarray(xTrack),np.asarray(yTrack)
NonLinearResiduals = np.sqrt((xSteps-xTrack)**2 + (ySteps-yTrack)**2)
plt.plot(steps,NonLinearResiduals,'.')
plt.plot(steps,CentroidResiduals,'.')
plt.show()

fig3, (ax31,ax32) = plt.subplots(1,2, figsize = (7,3))
ax31.contourf(x,y,zs[-1])
ax31.plot(xSteps[::15],ySteps[::15],'-.', label = 'Raw Data Path')
ax31.plot(cenXTrack[::15],cenYTrack[::15],'-.', label = 'Centroid Fit Path')
ax31.plot(xTrack[::15],yTrack[::15],'-.', label = 'Non Linear Least Squares Fit')
ax31.set_aspect('equal')
ax32.plot(steps,NonLinearResiduals,'.')
ax32.plot(steps,CentroidResiduals,'.')

plt.show()
