'''
This code is going to be used to simulate a particle moving in or around a map, it can have constraints like boxes as well
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
#============================================
#Functions
#============================================
def gauss2d(xCenter,yCenter,x,y,sigma):
    xcomp = ((x-xCenter)**2)/(2*sigma**2) 
    ycomp = ((y-yCenter)**2)/(2*sigma**2)
    
    return np.exp(-(xcomp+ycomp))

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
        returnedZs += 3*gauss2d(i,j,x,y,1)
        arrayZs.append(returnedZs)
    
    return arrayZs

#Fits using a centroid
def centroidFitter(threshold,imageSeries,x,y):
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
        
       
        
        xCoords.append(xcoord)
        yCoords.append(ycoord)


    return xCoords,yCoords

#My least squares algoritm
def leastSquaresFitter():






    


    

   





#============================================
#Code
#============================================
np.random.seed(0)
xBounds = (-25,25)
yBounds = (-25,25)
gridSize = (100,100)
xs,ys= (boxMapMaker(xBounds,yBounds,9))
xSteps,ySteps = walkerWithBox(150,0.5,xs,ys,.05,(0,0))

interactivePlotter(xs,ys,xSteps,ySteps)



zs = (dataSimulator(xBounds,yBounds,gridSize,(xSteps,ySteps)))
print(np.shape(zs))
#Plot counturf

x = np.linspace(xBounds[0],xBounds[1],gridSize[0])
y = np.linspace(yBounds[0],yBounds[1],gridSize[1])
x,y = np.meshgrid(x,y)

#Our centroid tracker
x = np.linspace(xBounds[0],xBounds[1],gridSize[0])
y = np.linspace(yBounds[0],yBounds[1],gridSize[1])
xTrack, yTrack = centroidFitter(1,zs,x,y)
trackedCoords = (zip(xTrack,yTrack))
print(xTrack)

plt.ion()
for i,j in zip(zs,trackedCoords):
    xcoord,ycoord = j
    plt.clf()
    plt.contourf(x,y,i)
    plt.colorbar()
    plt.plot(xcoord,ycoord,'o',color = 'orange')
    plt.draw()
    plt.pause(0.01)
plt.show()
#Residual
plt.ioff()

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
