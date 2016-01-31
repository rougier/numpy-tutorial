import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



def show(Z, shape, filename):
    Z = np.atleast_2d(Z)

    fig = plt.figure(figsize=(shape[1]/4.,shape[0]/4.), dpi=72)
    ax = plt.subplot(111)
    plt.imshow(Z, cmap='Purples', extent=[0,Z.shape[1],0,Z.shape[0]],
               vmin=0, vmax=max(1,Z.max()), interpolation='nearest', origin='upper')
    plt.xticks([]), plt.yticks([])
    plt.xlim(0,shape[1])
    plt.ylim(0,shape[0])

    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
        ax.spines[pos].set_alpha(.25)
        if Z.shape != shape:
            ax.spines[pos].set_linestyle('dashed')
            ax.spines[pos].set_alpha(.1)
    if Z.shape != shape:
        rect = Rectangle((0.01,0.01),Z.shape[1],Z.shape[0], 
                         zorder=+10,edgecolor='black', alpha=.25,facecolor='None')
        ax.add_patch(rect)
        ax.set_axisbelow(True)
    plt.savefig(filename,dpi=72)



Z1 = np.random.uniform(0,1,(9,5))
Z2 = np.ones((1,1))
Z3 = np.ones(Z1.shape)
show(Z1, Z1.shape, "../figures/broadcast-1.1.png")
show(Z2, Z1.shape, "../figures/broadcast-1.2.png")
show(Z3, Z1.shape, "../figures/broadcast-1.3.png")
show(Z1+Z2, Z1.shape, "../figures/broadcast-1.4.png")

Z2 = np.arange(9).reshape(9,1)
Z3 = np.repeat(Z2,5).reshape(9,5)
show(Z1, Z1.shape, "../figures/broadcast-2.1.png")
show(Z2, Z1.shape, "../figures/broadcast-2.2.png")
show(Z3, Z1.shape, "../figures/broadcast-2.3.png")
show(Z1+Z2, Z1.shape, "../figures/broadcast-2.4.png")

Z2 = np.arange(5).reshape(1,5)
#Z3 = np.zeros(Z1.shape)
Z3 = np.repeat(Z2,9).reshape(5,9).T
show(Z1, Z1.shape, "../figures/broadcast-3.1.png")
show(Z2, Z1.shape, "../figures/broadcast-3.2.png")
show(Z3, Z1.shape, "../figures/broadcast-3.3.png")
show(Z1+Z2, Z1.shape, "../figures/broadcast-3.4.png")

Z = np.zeros((9,5))
Z1 = np.arange(9).reshape(9,1)
Z2 = np.arange(5).reshape(1,5)
Z3 = np.repeat(Z1,5).reshape(9,5)
Z4 = np.repeat(Z2,9).reshape(5,9).T
show(Z1, Z.shape, "../figures/broadcast-4.1.png")
show(Z2, Z.shape, "../figures/broadcast-4.2.png")
show(Z3, Z.shape, "../figures/broadcast-4.3.png")
show(Z4, Z.shape, "../figures/broadcast-4.4.png")
show(Z1+Z2, Z.shape, "../figures/broadcast-4.5.png")


# def broadcast(Z1,Z2,Z3,Z4,Z5,filename):

#     filename ''.join(filename.split('.')[:-1])

#     show_array(Z1, filename+'.1.png')
#     show_array(Z1, filename+'.1.png')
#     show_array(Z1, filename+'.1.png')
#     show_array(Z1, filename+'.1.png')


#     def print_array(Z,ox,oy,C):
#         for x in range(Z.shape[1]):
#             for y in range(Z.shape[0]):
#                 if x >= C.shape[1] or y >= C.shape[0]:
#                     color = '.75'
#                     zorder=  -1
#                 else:
#                     color = 'k'
#                     zorder=  +1
#                 plt.text(ox+x+0.5, rows-0.5-oy-y, '%d' % Z[y,x],
#                          ha='center', va= 'center', size=24, color=color)
#                 rect = Rectangle((ox+x,rows-1+oy-y),1,1, zorder=zorder,edgecolor=color, facecolor='None')
#                 ax.add_patch(rect)
    
#     rows = 4
#     cols = 5*3 + (5-1)
#     fig = plt.figure(figsize=(cols,rows), dpi=72, frameon=False)
#     ax = plt.axes([0.05,0.05,.9,.9], frameon=False)
#     plt.xlim(0,cols), plt.xticks([])
#     plt.ylim(0,rows), plt.yticks([])
#     ox,oy = 0.0125, 0.0125
#     print_array(Z1,ox+0,oy,Z1)
#     plt.text(3.5, 2, '+', ha='center', va= 'center', size=48)
#     print_array(Z2,ox+4,oy,Z2)
#     plt.text(7.5, 2, '=', ha='center', va= 'center', size=48)
#     print_array(Z3,ox+8,oy,Z1)
#     plt.text(11.5, 2, '+', ha='center', va= 'center', size=48)
#     print_array(Z4,ox+12,oy,Z2)
#     plt.text(15.5, 2, '=', ha='center', va= 'center', size=48)
#     print_array(Z5,ox+16,oy,Z5)
# #    plt.savefig('../figures/%s' % name, dpi=32)
# #    plt.show()




# Z  = np.zeros((4,3))
# Z1 = np.repeat(np.arange(4)*10,3).reshape(4,3)
# Z2 = np.zeros((1,1))
# Z3 = Z1+Z2
# broadcast(Z1,Z2,Z+Z1,Z+Z2,Z1+Z2, "../figures/broadcast-1.png")


# Z  = np.zeros((4,3))
# Z1 = np.repeat(np.arange(4)*10,3).reshape(4,3)
# Z2 = np.resize(np.arange(3),(4,3))
# Z3 = Z1+Z2
# broadcast(Z1,Z2,Z+Z1,Z+Z2,Z1+Z2, "../figures/broadcast-2.png")


# Z  = np.zeros((4,3))
# Z1 = np.repeat(np.arange(4)*10,3).reshape(4,3)
# Z2 = np.arange(3).reshape(1,3)
# Z3 = Z1+Z2
# broadcast(Z1,Z2,Z+Z1,Z+Z2,Z1+Z2, "../figures/broadcast-3.png")


# Z  = np.zeros((4,3))
# Z1 = np.arange(4).reshape(4,1)*10
# Z2 = np.arange(3).reshape(1,3)
# Z3 = Z1+Z2
# broadcast(Z1,Z2,Z+Z1,Z+Z2,Z1+Z2, "../figures/broadcast-4.png")



"""
def show_array(Z1, Z2, name, a):
    Z1 = np.atleast_2d(Z1)
    rows,cols = Z2.shape
    fig = figure(figsize=(cols,rows), dpi=72, frameon=False)
    ax = axes([0,0,1,1], frameon=False)
    if a:
        imshow(Z2, cmap='Purples', extent=[0,Z2.shape[1],0,Z2.shape[0]], alpha=.5,
               vmin=-0.1, vmax=1, interpolation='nearest', origin='lower')
        hold(True)
    imshow(Z1, cmap='Purples', extent=[0,Z1.shape[1],0,Z1.shape[0]],
           vmin=-0.1, vmax=1, interpolation='nearest', origin='lower')
    xlim(0,Z2.shape[1]), xticks([])
    ylim(0,Z2.shape[0]), yticks([])
    savefig('../figures/%s' % name, dpi=16)
    #show()


rows,cols = 5, 9

Z1 = np.array([1.0])
Z2 = np.resize(Z1,(5,9))
show_array(Z1,Z2, 'broadcasts-1-before.png', False)
show_array(Z1,Z2, 'broadcasts-1-after.png', True)

Z1 = np.linspace(0,1,cols).reshape(1,cols)
Z2 = np.repeat(Z1,rows,axis=0)
show_array(Z1,Z2, 'broadcasts-2-before.png', False)
show_array(Z1,Z2, 'broadcasts-2-after.png', True)

Z1 = np.linspace(0,1,rows).reshape(rows,1)
Z2 = np.repeat(Z1,cols,axis=1)
show_array(Z1,Z2, 'broadcasts-3-before.png', False)
show_array(Z1,Z2, 'broadcasts-3-after.png', True)
"""
