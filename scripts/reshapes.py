import numpy as np
import matplotlib.pyplot as plt


def show_slice(Z, name):
    rows,cols = Z.shape
    fig = plt.figure(figsize=(cols/4.,rows/4.), dpi=72)
    ax = plt.subplot(111)
    A = np.arange(rows*cols).reshape(rows,cols)
    plt.imshow(Z, cmap='Purples', extent=[0,cols,0,rows],
               vmin=0, vmax=1, interpolation='nearest', origin='upper')
    plt.xticks([]), plt.yticks([])
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
        ax.spines[pos].set_alpha(.25)
    plt.savefig('../figures/%s' % name, dpi=72)
#    plt.show()


rows,cols = 3, 4

Z = np.zeros((rows,cols))+.1
Z[2,2] = 1
show_slice(Z, 'reshape-Z.png')

Z = Z.reshape(4,3)
show_slice(Z, 'reshape-Z-reshape(4,3).png')

Z = Z.reshape(12,1)
show_slice(Z, 'reshape-Z-reshape(12,1).png')

Z = Z.reshape(1,12)
show_slice(Z, 'reshape-Z-reshape(1,12).png')

Z = Z.reshape(6,2)
show_slice(Z, 'reshape-Z-reshape(6,2).png')

Z = Z.reshape(2,6)
show_slice(Z, 'reshape-Z-reshape(2,6).png')

