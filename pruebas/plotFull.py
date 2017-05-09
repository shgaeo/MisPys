#fuentes:
# http://stackoverflow.com/questions/13942956/disable-matplotlib-toolbar
# https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python/23755272#23755272?newreg=0cbc31d7e94e4fdf86a684c49057bc49
# http://stackoverflow.com/questions/16057869/setting-the-size-of-the-plotting-canvas-in-matplotlib

import numpy as np

import matplotlib as mpl 
#mpl.rcParams['toolbar'] = 'None'  ## remueve toolbar (el de arriba)

from matplotlib import pyplot as plt 
plt.switch_backend('TkAgg') # para permitir >>>orden fig.canvas.toolbar.pack_forget()


data = np.random.random((1080,1920))
#fig = plt.imshow(data,interpolation='nearest')
#fig.set_cmap('hot')


# load the image
img = plt.imread('thetamat.png')
## get the dimensions
#ypixels, xpixels, bands = img.shape
## get the size in inches
#dpi = 72.
#xinch = xpixels / dpi
#yinch = ypixels / dpi
## plot and save in the same size as the original
#fig = plt.figure(figsize=(xinch,yinch))

#fig=plt.figure(num=None, figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
fig=plt.figure()
ax = plt.axes([0., 0., 1., 1.], frameon=False, xticks=[],yticks=[])
#ax.imshow(img, interpolation='none')
ax.imshow(data, interpolation='none')
#plt.imshow(data, interpolation='none')
#ax.format_coord = lambda x, y: ''
fig.canvas.toolbar.pack_forget() ########################################
####################### revisa http://stackoverflow.com/questions/17573346/remove-status-bar-of-a-matplotlib-figure
####################### según este tipo se puede eliminar la barra de abajo usando ese comando pero debo estar en "TkAgg backend" Checa esto

plt.axis('off')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()      


plt.show()

#plt.savefig('mpl_logo.png', transparent=True)

