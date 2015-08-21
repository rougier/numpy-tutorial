from pylab import *
from matplotlib.patches import Rectangle


def show_slice(Z, name):
    Z = np.atleast_2d(Z)
    rows,cols = Z.shape
    fig = figure(figsize=(cols,rows), dpi=72, frameon=False)
    #ax = subplot(111, frameon=False)
    ax = axes([0,0,1,1], frameon=False)
    A = np.arange(rows*cols).reshape(rows,cols)
    imshow(Z, cmap='Purples', extent=[0,cols,0,rows],
           vmin=0, vmax=max(1,Z.max()), interpolation='nearest', origin='upper')
    #xticks(1.05+np.arange(cols-1),[]), yticks(1+np.arange(rows-1),[])
    xticks([]), yticks([])
    #ax.grid(which='major', axis='x', linewidth=1.5, linestyle='-', color='w')
    #ax.grid(which='major', axis='y', linewidth=1.5, linestyle='-', color='w')
    #ax.tick_params(axis='x', colors='w')
    #ax.tick_params(axis='y', colors='w')
    savefig('../figures/%s' % name, dpi=16)
    #show()


rows,cols = 5, 9

Z = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])
Z += 0.1
show_slice(Z, 'create-list-1.png')

Z = np.zeros(cols)+.1
show_slice(Z, 'create-zeros-1.png')

Z = np.ones(cols)+.1
show_slice(Z, 'create-ones-1.png')

Z = np.arange(cols)
show_slice(Z, 'create-arange-1.png')

Z = np.random.uniform(0,1,cols)
show_slice(Z, 'create-uniform-1.png')

Z = np.zeros((rows,cols))+.1
show_slice(Z, 'create-zeros-2.png')

Z = np.ones((rows,cols))+.1
show_slice(Z, 'create-ones-2.png')

Z = np.array([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
              [0.,0.,0.,0.,0.,0.,0.,0.,0.],
              [0.,0.,0.,0.,0.,0.,0.,0.,0.],
              [0.,0.,0.,0.,0.,0.,0.,0.,0.],
              [0.,0.,0.,0.,0.,0.,0.,0.,0.]])
Z += 0.1
show_slice(Z, 'create-list-2.png')

Z = np.arange(rows*cols).reshape(rows,cols)
show_slice(Z, 'create-arange-2.png')

Z = np.random.uniform(0,1,(rows,cols))
show_slice(Z, 'create-uniform-2.png')
