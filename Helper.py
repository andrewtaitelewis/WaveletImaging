'''
This file contains all the helper methods
'''
#===================================
#Importing the necessary modules
#===================================
import numpy as np
import random
#Methods
#===================================
#AutoCorrelation.py
#===================================
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
#One dimensional autocorrelation function
def autocorrelationFunction(tau,signal):
    ''' returns the autocorrelation function at all later times of t'''
    average = np.average(signal)**2     #The bottom of our equation
    topAverage = []
    lastIndex = len(signal) -1  #The last point in the array
    t = 0
    while t + tau <= lastIndex: #Goes through the entire array
        topAverage.append(signal[t]*signal[t+tau])
        t += 1
    
    topAverage = np.average(topAverage)
    return (topAverage/average) + 1
#Produces a sin function hidden in noise
def randomizedSin(length):
    returned = []
    for i in range(length):
        returned.append(np.sin(i) + 5*np.random.rand())
    return returned
#random walkers
def xRandomWalker(numberOfWalkers,radius,steps,startingPoint):
    '''
    A module that produces the path of x random walkers
    Parameters:
    -----------
    numberOfWalkers, int: number of random walkers
    radius, int: the radius a random walker can just around
    steps, int: the number of steps the walkers will take
    startingpoint, (float*float): a tuple that has the starting point of all the walkers
    Returns:
    --------
    walkerStepsArray = array of arrays of tuple arrays
    ((minx,maxX),(miny,maxy)) = tuples useful in constructing the gird
    '''
    walkerStepsArray = []   #Initalize the array of walker steps
    minX,maxX,minY,maxY = 0,0,0,0
    for i in range(numberOfWalkers):
        currentX,currentY = startingPoint
        currentXArray = [startingPoint[0]]
        currentYArray = [startingPoint[1]]
        for j in range(steps):
            #Get the random steps
            xChange = (np.random.randint(-1,2))*radius
            yChange = (np.random.randint(-1,2))*radius
            #Move the walker
            currentX,currentY = currentX + xChange, currentY + yChange
            #Add the steps to the walker
            currentXArray.append(currentX)
            currentYArray.append(currentY)
        walkerStepsArray.append((currentXArray,currentYArray))
        #Check to see if the max and min coords have changed
        if min(currentYArray) < minY:
            minY = min(currentYArray)
        if max(currentYArray) > maxY:
            maxY = max(currentYArray)
        if min(currentXArray) < minX:
            minX = min(currentXArray)
        if max(currentXArray) > maxX:
            maxX = max(currentXArray)
    return walkerStepsArray, ((minX,maxX),(minY,maxY))
#Random walkers with flow
def flowXRandomWaler(numberOfWalkers,radius,steps,startingPoint,flowStrength):
    '''
    A module that produces the path of x random walkers
    Parameters:
    -----------
    numberOfWalkers, int: number of random walkers
    radius, int: the radius a random walker can just around
    steps, int: the number of steps the walkers will take
    startingpoint, (float*float): a tuple that has the starting point of all the walkers
    Returns:
    --------
    walkerStepsArray = array of arrays of tuple arrays
    ((minx,maxX),(miny,maxy)) = tuples useful in constructing the gird
    '''
    
    walkerStepsArray = []   #Initalize the array of walker steps
    minX,maxX,minY,maxY = 0,0,0,0
    for i in range(numberOfWalkers):
        currentX,currentY = startingPoint
        currentXArray = [startingPoint[0]]
        currentYArray = [startingPoint[1]]
        for j in range(steps):
            #Get the random steps
            xChange = (np.random.randint(-1,2))*radius
            yChange = (np.random.randint(-1,2))*radius
            #Move the walker
            currentX,currentY = currentX + xChange, currentY + yChange
            #Add the flow strength
            currentX,currentY = currentX + flowStrength, currentY +flowStrength
            #Add the steps to the walker
            currentXArray.append(currentX)
            currentYArray.append(currentY)
            

        walkerStepsArray.append((currentXArray,currentYArray))
        #Check to see if the max and min coords have changed
        if min(currentYArray) < minY:
            minY = min(currentYArray)
        if max(currentYArray) > maxY:
            maxY = max(currentYArray)
        if min(currentXArray) < minX:
            minX = min(currentXArray)
        if max(currentXArray) > maxX:
            maxX = max(currentXArray)
    return walkerStepsArray, ((minX,maxX),(minY,maxY))
#Temporal Image Correlation Function
def tic(startingFrame,imageSeries,tauRange):
    '''
    Temportal image correlation as defined in Kolin et Al.
    Parmaters:
    ----------
    startingFrame: integer of the starting frame
    imageSeries: array of 2d image arrays
    tauRange: int, how many images we want to go into the array, same size as returnedArray
    Returns:
    --------
    correlationArray: an array of the correlation coefficients
    '''
    firstImage = imageSeries[startingFrame]
    firstImageAverage = np.average(firstImage.flatten())    #Gets the average intensity of the first image
    bounds = np.shape((firstImage))
    
    xs,ys = bounds  #assign the bounds
    deltaFirstImage = firstImage - firstImageAverage    #Subtract the first image
    returnedArray = []
    for i in range(tauRange):
        tauImage = imageSeries[i]   #The image we are working with
        tauAverage = np.average(tauImage.flatten()) #The average intensity over the image
        deltaTau = tauImage - tauAverage    #computing delta(b)
        #Now we multiply the two deltas
        multiplied = np.asarray(deltaFirstImage)*np.asarray(deltaTau)
        multiAverage = np.average(multiplied)


        returnedArray.append(multiAverage/(firstImageAverage * tauAverage))
    
    
            
    return  returnedArray

#===================================
#DataSimulator
#===================================
def brownianChange(xPrev,yPrev):
    '''
    Updates the x,y coordiantes of a given point
    Params:
    -------
    xPrev = previous x coordinate
    yPrev = previous y coordinnte
    Returns:
    --------
    xNew = new x coordinate
    yNew = new y coordinate
    '''
    xStep = np.random.normal(0.0,1.0)
    yStep = np.random.normal(0.0,1.0)
    return xPrev+xStep, yPrev+yStep

def probMover(xCoords,yCoords, xCur,yCur):
    '''
    Params:
    -------
    xCoords = max x value
    yCoords = max y value
    xCur = current x value
    yCur = current y value
    '''
    def move(xCur,yCur):
        if np.random.rand() < .5:
            xCur = xCur - 1
        else:
            xCur = xCur + 1
        if np.random.rand() <.5:
            yCur = yCur -1
        else:
            yCur = yCur + 1
        #Now check if they're valid moves
        if yCur < 0:
            yCur = yCur +1
        if yCur >= yCoords:
            yCur = yCur -1
        if xCur <0:
            xCur = xCur +1
        if xCur >= xCoords:
            xCur = xCur - 1
        return xCur,yCur
    #Center 
    xCenter = xCoords/2.0
    yCenter = yCoords/2.0
    #Normalize so all values are bettween 0 and 1
    normalizer = np.sqrt(xCenter**2 + yCenter**2)
    distance = np.sqrt((xCur-xCenter)**2 +(yCur-yCenter)**2)
    probabilityThreshold = distance/normalizer
    #Now lets try and move something
    
    if np.random.rand() < probabilityThreshold:
        xCur,yCur = move(xCur,yCur)
    if np.random.rand() < 0.3:
        xCur,yCur = move(xCur,yCur)
    
    return xCur,yCur

def metropolisHasting(Func, currentPosition, candidatePosition):
    '''
    A walker initalzied with the metropolis hastings algorithm
    Params:
    -------
    Func: function of acceptance 
    currentPosition: array of where the walker is, must be passable to our function
    candidatePosition: array of where the walker might want to go, must be passable to our function
    Returns:
    --------
    '''
    print(Func(currentPosition))

#===================================
#Particle Simulator
#===================================
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

#===================================
#TryingSomethingOut
#===================================
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
#Two Dimensional Mexican Hat Wavelet
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

    