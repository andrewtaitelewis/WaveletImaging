'''
Simulating data to test my code 
'''
#Importing Packages
import numpy as np
import random
import matplotlib.pyplot as plt
#============================================================
#Helper Function
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

print(np.random.rand())
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

metropolisHasting(add,[1,2],4)

#=============================================================
#Trying out my walker
'''
plt.ion()   #Turns interactive mode on
initX = 0
initY = 0
for i in range(50):
    plt.plot(initX,initY,'.')
    plt.xlim(-20,20)    #Xlim and yLim make it so the window doensn't move
    plt.ylim(-20,20)
    initX,initY = brownianChange(initX,initY)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()    
'''
#Okay now for something a little more daunting


#Code
#Making the grid
xCoords = 50
yCoords = 50
#XCoordxYcoord array of zeros
grid = []
for i in range(xCoords):
    grid.append(np.zeros(yCoords))

#Now we want to initalize it with a random int from 0-2
print(grid[0][2])
for i in range(xCoords):
    for j in range(yCoords):
        grid[i][j] = random.randint(0,2)
#Now plot our initial grid
plt.ion()
runs = 2000
for t in range(runs):
    for i in range(xCoords):
        for j in range(yCoords):
            for k in range(int(grid[i][j])):
                newI,newJ = probMover(xCoords,yCoords,i,j)
                if newI == i and newJ == j:
                    continue
                else:
                    grid[i][j] = grid[i][j] - 1
                    grid[newI][newJ] += 1
    if t%50 == 0:

        plt.imshow(grid,vmax = 20)
        plt.colorbar()
        plt.draw()
        plt.pause(0.01)
        plt.clf()
   

    
plt.ioff()
plt.imshow(grid)
plt.colorbar()
plt.show()



