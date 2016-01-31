import numpy as np
import matplotlib.pyplot as plt


def show_array(Z, name):

    Z = np.atleast_2d(Z)
    rows,cols = Z.shape
    fig = plt.figure(figsize=(cols/4.,rows/4.), dpi=72)
    ax = plt.subplot(111)
    #plt.imshow(Z, cmap='Purples', extent=[0,cols,0,rows],
    #           vmin=-.2, vmax=1, interpolation='nearest', origin='upper')
    plt.imshow(Z, cmap='Purples', extent=[0,cols,0,rows],
               interpolation='nearest', origin='upper')
    plt.xticks([]), plt.yticks([])
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
        ax.spines[pos].set_alpha(.25)
    plt.savefig('../figures/%s' % name, dpi=72)
    #plt.show()


rows,cols = 5, 9

Z1 = np.linspace(0,1,rows*cols).reshape(rows,cols)
show_array(Z1, 'ops-where-before.png')
Z2 = np.where(Z1 > 0.5, 0, 1)
show_array(Z2, 'ops-where-after.png')

Z1 = np.linspace(0,1,rows*cols).reshape(rows,cols)
show_array(Z1, 'ops-maximum-before.png')
Z2 = np.maximum(Z1, 0.5)
show_array(Z2, 'ops-maximum-after.png')

Z1 = np.linspace(0,1,rows*cols).reshape(rows,cols)
show_array(Z1, 'ops-minimum-before.png')
Z2 = np.minimum(Z1, 0.5)
show_array(Z2, 'ops-minimum-after.png')

Z1 = np.linspace(0,1,rows*cols).reshape(rows,cols)
show_array(Z1, 'ops-sum-before.png')
Z2 = Z1.sum(axis=0)
show_array(Z2, 'ops-sum-after.png')

