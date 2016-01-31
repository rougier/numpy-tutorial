import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z  = np.zeros((32,32), dtype=int)
Z[1:4,1:4] = [[0,0,1],
              [1,0,1],
              [0,1,1]]

for i in range(4*20):
    iterate(Z)

size = 4*np.array(Z.shape)
dpi = 72.0
figsize= size[1]/float(dpi),size[0]/float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
fig.add_axes([0.025, 0.025, .95, .95], frameon=True)
plt.imshow(Z,interpolation='nearest', cmap=plt.cm.Purples)
plt.xticks([]), plt.yticks([])
plt.savefig('../figures/exercice-1.png')
plt.show()
