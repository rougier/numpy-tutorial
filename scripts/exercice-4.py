'''
Reaction Diffusion : Gray-Scott model

References:
----------
Complex Patterns in a Simple System
John E. Pearson, Science 261, 5118, 189-192, 1993.

Encode movie
------------

ffmpeg -r 30 -i "tmp-%03d.png" -c:v libx264 -crf 23 -pix_fmt yuv420p bacteria.mp4
'''
import numpy as np
import matplotlib.pyplot as plt

n,k  = 100, 4
Z = np.zeros((n+2,2*n+2))
dt = 0.05

plt.ion()

size = 3*np.array(Z.shape)
dpi = 72.0
figsize= size[1]/float(dpi),size[0]/float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
im = plt.imshow(Z, interpolation='bicubic', cmap=plt.cm.hot)
plt.xticks([]), plt.yticks([])

for i in range(50000):
    L = (                 Z[0:-2,1:-1] +
         Z[1:-1,0:-2] - 4*Z[1:-1,1:-1] + Z[1:-1,2:] +
                          Z[2:  ,1:-1] )
    Z[1:-1,1:-1] += k*L*dt
    Z[ n/2-20:n/2+20, n-5:n-1] =  1
    Z[ n/2-20:n/2+20, n+1:n+5] = -1

    if i % 30 == 0:
        im.set_data(Z)
        im.set_clim(vmin=Z.min(), vmax=Z.max())
        plt.draw()
        # To make movie
        # plt.savefig("./tmp/tmp-%03d.png" % (i/10) ,dpi=dpi)

plt.ioff()
# plt.savefig("../figures/zebra.png",dpi=dpi)
# plt.savefig("../figures/bacteria.png",dpi=dpi)
plt.savefig("../figures/diffusion.png",dpi=dpi)
plt.show()
