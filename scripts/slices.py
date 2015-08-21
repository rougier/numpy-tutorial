from pylab import *
from matplotlib.patches import Rectangle


def show_slice(Z, name):
    rows,cols = Z.shape
    fig = figure(figsize=(cols/4.,rows/4.), dpi=72)
    ax = subplot(111)
    A = np.arange(rows*cols).reshape(rows,cols)
    imshow(Z, cmap='Purples', extent=[0,cols,0,rows],
           vmin=0, vmax=1, interpolation='nearest', origin='upper')
    xticks([]), yticks([])
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
        ax.spines[pos].set_alpha(.25)
    savefig('../figures/%s' % name, dpi=72)
    #show()



rows,cols = 5, 9

Z = np.zeros((rows,cols))+.1
show_slice(Z, 'slice-Z.png')

Z = np.zeros((rows,cols))+.1
Z[...] = 1
show_slice(Z, 'slice-Z[...].png')

Z = np.zeros((rows,cols))+.1
Z[:,::2] = 1
show_slice(Z, 'slice-Z[:,::2].png')

Z = np.zeros((rows,cols))+.1
Z[::2,:] = 1
show_slice(Z, 'slice-Z[::2,:].png')

Z = np.zeros((rows,cols))+.1
Z[1,1] = 1
show_slice(Z, 'slice-Z[1,1].png')

Z = np.zeros((rows,cols))+.1
Z[:,0] = 1
show_slice(Z, 'slice-Z[:,0].png')

Z = np.zeros((rows,cols))+.1
Z[0,:] = 1
show_slice(Z, 'slice-Z[0,:].png')

Z = np.zeros((rows,cols))+.1
Z[2:,2:] = 1
show_slice(Z, 'slice-Z[2:,2:].png')

Z = np.zeros((rows,cols))+.1
Z[:-2,:-2] = 1
show_slice(Z, 'slice-Z[:-2,:-2].png')

Z = np.zeros((rows,cols))+.1
Z[2:4,2:4] = 1
show_slice(Z, 'slice-Z[2:4,2:4].png')

Z = np.zeros((rows,cols))+.1
Z[::2,::2] = 1
show_slice(Z, 'slice-Z[::2,::2].png')


Z = np.zeros((rows,cols))+.1
Z[3::2,3::2] = 1
show_slice(Z, 'slice-Z[3::2,3::2].png')

