import numpy as np
import matplotlib.pyplot as plt

#
# Routines for reading and plotting images
#
def im5k_show(k):
  fname = "/data/images_npy/" + str(k) + ".npy"
  d = np.load(fname)
  print("  Min value:", d.min())
  print("  Max value:", d.max())
  e = 0.5 + 0.5*d
  plt.imshow(np.transpose(e, (1,2,0)))
  plt.show()


#
# This routine updates the list of top and bottom 10 images
#
def topbot_update(tr,ti,tc,br,bi,bc,rd,ii):
  #  tr - top responses, largest is first
  #  ti - top image indices
  #  tc - top coordinate as tuple
  #  br - bot responses, smallest is first
  #  bi - bot image indices
  #  bc - bot coordinate as tuple
  #  rd - response data [spatial][spatial] 
  #  ii   - image index
  
  # Compute the indices (i0,i1) for the max and min in the spatial sheet
  shn = len(rd)  # width of sheet
  imax = int(rd.argmax())
  imax_i0 = int(imax / shn)
  imax_i1 = imax % shn
  
  imin = int(rd.argmin())
  imin_i0 = int(imin / shn)
  imin_i1 = imin % shn
  
  rmax = float(rd[imax_i0][imax_i1])  # Get max, min values using the indices
  rmin = float(rd[imin_i0][imin_i1])
  
  nn = len(tr)    # E.g., this is 10 if we are keeping top 10
  for i in range(nn-1,-1,-1):
    if (ti[i] != -1):
      break
  
  if (i == 0) & (ti[i] == -1):  # The lists must both be empty
    ti[0] = ii
    tr[0] = rmax
    tc[0] = (imax_i0, imax_i1)
    bi[0] = ii
    br[0] = rmin
    bc[0] = (imin_i0, imin_i1)
  else:           # 'i' now points to the smallest entry in the list
    k = i  
    while (rmax > tr[k]):  # while 'rmax' is larger than stored value
      if (k < nn-1):       # Move this value down in the list
        tr[k+1] = tr[k]
        ti[k+1] = ti[k]
        tc[k+1] = tc[k]
      k -= 1               # Move 'k' up
      if (k < 0):
        break
    
    if (k+1 < nn):
      tr[k+1] = rmax
      ti[k+1] = ii
      tc[k+1] = (imax_i0, imax_i1)
    
    k = i  
    while (rmin < br[k]):  # while 'rmin' is smaller than stored value
      if (k < nn-1):       # Move this value down in the list
        br[k+1] = br[k]
        bi[k+1] = bi[k]
        bc[k+1] = bc[k]
      k -= 1               # Move 'k' up
      if (k < 0):
        break
    
    if (k+1 < nn):
      br[k+1] = rmin
      bi[k+1] = ii
      bc[k+1] = (imin_i0, imin_i1)




# *** START HERE ***

#
# (0) Initialize the model, setting 'm' and 'zn' for use below
#
# mod = models.alexnet(pretrained=True)
# li_conv1 =  0   # Store layer indices ("li_...") for particular layers
# li_conv2 =  3
# li_conv3 =  6
# li_conv4 =  8
# li_conv5 = 10
# li_fc6   =  1
# li_fc7   =  4
# li_fc8   =  6
# m = mod.features[:li_conv5+1]   # Or change 'li_conv5' to any other layer HERE
# zn = m[li_conv5].out_channels   #   and HERE

# #  *** OR, use VGG16 here ***

# mod = models.vgg16(pretrained=True)
# li_conv1  =  0   # Store layer indices ("li_...") for particular layers
# li_conv2  =  2
# li_conv3  =  5
# li_conv4  =  7
# li_conv5  = 10
# li_conv6  = 12
# li_conv7  = 14
# li_conv8  = 17
# li_conv9  = 19
# li_conv10 = 21
# li_conv11 = 24
# li_conv12 = 26
# li_conv13 = 28
# m = mod.features[:li_conv7+1]   # Or change 'li_conv7' to any other layer HERE
# zn = m[li_conv7].out_channels   #   and HERE

# #
# #  Initialize data structures to hold top/bot images, responses, coordinates
# #
# nn = 10   # Top, Bottom 10
# top_r = np.empty((zn,nn), dtype='float32')  # Highest 'nn' responses
# bot_r = np.empty((zn,nn), dtype='float32')  # Lowest 'nn' responses
# top_i = np.full((zn,nn), -1, dtype='int')   # Image indices for top
# bot_i = np.full((zn,nn), -1, dtype='int')   # Image indices for bot
# top_c = []  # Empty list to hold coordinates
# for i in range(zn):
#   top_c.append([None]*nn)

# bot_c = []  # Empty list to hold coordinates
# for i in range(zn):
#   bot_c.append([None]*nn)

# #
# # In a loop,
# #   (1) read a batch of images (from Dean's text format, 227 x 227 pix)
# #   (2) Run them through the model
# #   (3) For each unit in layer, track top 10 and bottom 10 images
# #
# xn = 227       # My input images are this many pixels wide and tall
# nstim = 50     # Number of images in batch (arbitrary number)

# for i in range(1000):   # In 1000 batches (of 50 images each)
#   print(i)

#   # Get the indices of the images (e.g., 0, 1, 2, ...)
#   imgind = np.arange(i*nstim,(i+1)*nstim)         # Set image indices
#   d = np.empty((nstim,3,xn,xn), dtype='float32')  # Empty array to hold images
#   for j in range(nstim):
#     k = imgind[j]    # index for this image
#     fname = "/data/images_npy/" + str(k) + ".npy" # Image filename, eg "123.npy"
#     d[j] = np.load(fname)
  
#   print("  Read",nstim,"images, from",imgind[0])
#   #d.shape  # Check dimensions of 'd'
  
#   # Run the batch of images
#   tt = torch.tensor(d)   # Convert to tensor format
#   r = m.forward(tt)      # Run to get responses, r[stim, zfeature, x, y]
#   shn0 = r.shape[2]      # length of 1st spatial grid axis
#   shn1 = r.shape[3]      # length of 2nd spatial grid axis
  
#   #
#   #  For each unit in layer, track top 10 and bottom 10 images across batches
#   #
#   for zi in range(zn):
#     #print(zi)
#     for si in range(nstim):
#       ii = imgind[si]
#       topbot_update(top_r[zi],top_i[zi],top_c[zi],
#                     bot_r[zi],bot_i[zi],bot_c[zi],r[si][zi],ii)



#
#  When the loop over all images is finished, save the following information:
#
# np.save("n03_stat_conv12_t10_r",top_r)   # Top 10 responses
# np.save("n03_stat_conv12_t10_i",top_i)   # Top 10 image indices
# np.save("n03_stat_conv12_b10_r",bot_r)   # Bottom 10 responses
# np.save("n03_stat_conv12_b10_i",bot_i)   # Bottom 10 image indices

# import pickle
# with open("n03_stat_conv12_t10_c.txt","wb") as fp:  # The x,y grid coords.
#   pickle.dump(top_c, fp)                            #   for the particular unit

# with open("n03_stat_conv12_b10_c.txt","wb") as fp:  # Same for bottom 10.
#   pickle.dump(bot_c, fp)

# #
# #  *** FINISHED ***
# #


# #
# #  Load the output back in, to check that it was saved
# #
# top_r = np.load("npy_stat_conv5_t10_r.npy")
# top_i = np.load("npy_stat_conv5_t10_i.npy")
# bot_r = np.load("npy_stat_conv5_b10_r.npy")
# bot_i = np.load("npy_stat_conv5_b10_i.npy")

# with open("npy_stat_conv5_t10_c.txt","rb") as fp:
#   top_c = pickle.load(fp)

# with open("npy_stat_conv5_b10_c.txt","rb") as fp:
#   bot_c = pickle.load(fp)

# for i in top_i[0]:
#   im5k_show(i)

def im5k_draw_box(d,i0,j0,w):
  #      d  - numpy array [3][xn][xn] to over-write
  # (i0,j0) - initial point (pix)
  #      w  - size of box (pix)
  print("     box  (",i0,",",j0,")  wid",w)
  xn = len(d[0])
  for i in range(i0,i0+w):
    if (i >= 0) & (i <  xn):
      if (i == i0) | (i == i0+w-1):
        for j in range(j0,j0+w):
          if (j >= 0) & (j <  xn):
            d[0][i][j] = 1.0
            d[1][i][j] = 0.0
            d[2][i][j] = 0.0
      else:
        if (j0 >= 0) & (j0 <  xn):
          d[0][i][j0] = 1.0
          d[1][i][j0] = 0.0
          d[2][i][j0] = 0.0
        if (j0+w-1 >= 0) & (j0+w-1 <  xn):
          d[0][i][j0+w-1] = 1.0
          d[1][i][j0+w-1] = 0.0
          d[2][i][j0+w-1] = 0.0


def im5k_show_box(k, i0, j0, w, title):
  #
  #       k  - Index of image
  #  (i0,j0) - lower left corner of box (pix)
  #       w  - width of box (pix)
  #
  fname = "/data/images_npy/" + str(k) + ".npy"
  d = np.load(fname)
  dmin = d.min()
  dmax = d.max()
  d = d - dmin
  d = d / (dmax - dmin)
  im5k_draw_box(d,i0,j0,w)
  plt.imshow(np.transpose(d, (1,2,0)))
  plt.title(title)
  plt.show()


def topbot_show(ti,tr,tc,bi,br,bc,rfpd,k):
  #
  #  ti - [units][10]  top image indices
  #  tr - [units][10]  top responses
  #  tc - [units][10]  tuples of spatial coords
  #  bi - [units][10]  bot image indices
  #  br - [units][10]  bot responses
  #  bc - [units][10]  tuples of spatial coords
  #  rfpd - RF parameter dictionary
  #  k  - unit index
  #
  sz = rfpd['size']
  x0 = rfpd['x0']
  dx = rfpd['stride']
  print("  rfdp (sz, x0, dx):",sz,x0,dx)
  
  n = len(ti[k])
  for i in range(n):
    j = ti[k][i]
    print("   unit_xy= ",tc[k][i])
    i0 = x0 + dx * tc[k][i][0]
    j0 = x0 + dx * tc[k][i][1]
    print("  Top",i,"image",j," response =",tr[k][i])
    title = "Unit " + str(k) + "  Top " + str(i+1)
    im5k_show_box(j,i0,j0,sz,title)
  
  for i in range(n):
    j = bi[k][i]
    i0 = x0 + dx * bc[k][i][0]
    j0 = x0 + dx * bc[k][i][1]
    print("  Bottom",i,"image",j," response =",br[k][i])
    title = "Unit " + str(k) + "  Bot " + str(i+1)
    im5k_show_box(j,i0,j0,sz,title)


### AlexNet at stim size = 227 pix
a227 = {'conv1':{'i':  0, 'x0':  -2, 'stride':  4, 'size':   11, 'cnt':   56},
        'relu1':{'i':  1, 'x0':  -2, 'stride':  4, 'size':   11, 'cnt':   56},
        'pool1':{'i':  2, 'x0':  -2, 'stride':  8, 'size':   19, 'cnt':   27},
        'conv2':{'i':  3, 'x0': -18, 'stride':  8, 'size':   51, 'cnt':   27},
        'relu2':{'i':  4, 'x0': -18, 'stride':  8, 'size':   51, 'cnt':   27},
        'pool2':{'i':  5, 'x0': -18, 'stride': 16, 'size':   67, 'cnt':   13},
        'conv3':{'i':  6, 'x0': -34, 'stride': 16, 'size':   99, 'cnt':   13},
        'relu3':{'i':  7, 'x0': -34, 'stride': 16, 'size':   99, 'cnt':   13},
        'conv4':{'i':  8, 'x0': -50, 'stride': 16, 'size':  131, 'cnt':   13},
        'relu4':{'i':  9, 'x0': -50, 'stride': 16, 'size':  131, 'cnt':   13},
        'conv5':{'i': 10, 'x0': -66, 'stride': 16, 'size':  163, 'cnt':   13},
        'relu5':{'i': 11, 'x0': -66, 'stride': 16, 'size':  163, 'cnt':   13},
        'pool5':{'i': 12, 'x0': -60, 'stride': 30, 'size':  193, 'cnt':    6}}

### AlexNet at stim size = 224 pix (not used here)
a224 = {'conv1':{'i':  0, 'x0':  -2, 'stride':  4, 'size':   11, 'cnt':   55},
        'relu1':{'i':  1, 'x0':  -2, 'stride':  4, 'size':   11, 'cnt':   55},
        'pool1':{'i':  2, 'x0':  -2, 'stride':  8, 'size':   19, 'cnt':   27},
        'conv2':{'i':  3, 'x0': -18, 'stride':  8, 'size':   51, 'cnt':   27},
        'relu2':{'i':  4, 'x0': -18, 'stride':  8, 'size':   51, 'cnt':   27},
        'pool2':{'i':  5, 'x0': -18, 'stride': 16, 'size':   67, 'cnt':   13},
        'conv3':{'i':  6, 'x0': -34, 'stride': 16, 'size':   99, 'cnt':   13},
        'relu3':{'i':  7, 'x0': -34, 'stride': 16, 'size':   99, 'cnt':   13},
        'conv4':{'i':  8, 'x0': -50, 'stride': 16, 'size':  131, 'cnt':   13},
        'relu4':{'i':  9, 'x0': -50, 'stride': 16, 'size':  131, 'cnt':   13},
        'conv5':{'i': 10, 'x0': -66, 'stride': 16, 'size':  163, 'cnt':   13},
        'relu5':{'i': 11, 'x0': -66, 'stride': 16, 'size':  163, 'cnt':   13},
        'pool5':{'i': 12, 'x0': -60, 'stride': 30, 'size':  193, 'cnt':    6}}

#
#  Middle unit center pixel
#    Conv2  -18 + (27-1)/2 * stride + 
#
def center_unit_midpix(a):
  #
  #  a - parameter array for layer configuration, e.g. 'a' and 'v' from above
  #
  for lay in a:
    d = a[lay]  # Get dictionary for this layer
    cp = d['x0'] + int((d['cnt'] - 1)/2) * d['stride'] + (d['size']-1)/2
    print('  ' + lay + ' center pix:',cp)


### VGG16 at stim size = 227 pix
v227 = {'conv1' :{'i':  0, 'x0': -1, 'stride':  1, 'size':    3, 'cnt':   227},
        'relu1' :{'i':  1, 'x0': -1, 'stride':  1, 'size':    3, 'cnt':   227},
        'conv2' :{'i':  2, 'x0': -2, 'stride':  1, 'size':    5, 'cnt':   227},
        'relu2' :{'i':  3, 'x0': -2, 'stride':  1, 'size':    5, 'cnt':   227},
        'pool2' :{'i':  4, 'x0': -2, 'stride':  2, 'size':    6, 'cnt':   113},
        'conv3' :{'i':  5, 'x0': -4, 'stride':  2, 'size':   10, 'cnt':   113},
        'relu3' :{'i':  6, 'x0': -4, 'stride':  2, 'size':   10, 'cnt':   113},
        'conv4' :{'i':  7, 'x0': -6, 'stride':  2, 'size':   14, 'cnt':   113},
        'relu4' :{'i':  8, 'x0': -6, 'stride':  2, 'size':   14, 'cnt':   113},
        'pool4' :{'i':  9, 'x0': -6, 'stride':  4, 'size':   16, 'cnt':    56},
        'conv5' :{'i': 10, 'x0':-10, 'stride':  4, 'size':   24, 'cnt':    56},
        'relu5' :{'i': 11, 'x0':-10, 'stride':  4, 'size':   24, 'cnt':    56},
        'conv6' :{'i': 12, 'x0':-14, 'stride':  4, 'size':   32, 'cnt':    56},
        'relu6' :{'i': 13, 'x0':-14, 'stride':  4, 'size':   32, 'cnt':    56},
        'conv7' :{'i': 14, 'x0':-18, 'stride':  4, 'size':   40, 'cnt':    56},
        'relu7' :{'i': 15, 'x0':-18, 'stride':  4, 'size':   40, 'cnt':    56},
        'pool7' :{'i': 16, 'x0':-18, 'stride':  8, 'size':   44, 'cnt':    28},
        'conv8' :{'i': 17, 'x0':-26, 'stride':  8, 'size':   60, 'cnt':    28},
        'relu8' :{'i': 18, 'x0':-26, 'stride':  8, 'size':   60, 'cnt':    28},
        'conv9' :{'i': 19, 'x0':-34, 'stride':  8, 'size':   76, 'cnt':    28},
        'relu9' :{'i': 20, 'x0':-34, 'stride':  8, 'size':   76, 'cnt':    28},
        'conv10':{'i': 21, 'x0':-42, 'stride':  8, 'size':   92, 'cnt':    28},
        'relu10':{'i': 22, 'x0':-42, 'stride':  8, 'size':   92, 'cnt':    28},
        'pool10':{'i': 23, 'x0':-42, 'stride': 16, 'size':  100, 'cnt':    14},
        'conv11':{'i': 24, 'x0':-58, 'stride': 16, 'size':  132, 'cnt':    14},
        'relu11':{'i': 25, 'x0':-58, 'stride': 16, 'size':  132, 'cnt':    14},
        'conv12':{'i': 26, 'x0':-74, 'stride': 16, 'size':  164, 'cnt':    14},
        'relu12':{'i': 27, 'x0':-74, 'stride': 16, 'size':  164, 'cnt':    14},
        'conv13':{'i': 28, 'x0':-90, 'stride': 16, 'size':  196, 'cnt':    14},
        'relu13':{'i': 29, 'x0':-90, 'stride': 16, 'size':  196, 'cnt':    14},
        'pool13':{'i': 30, 'x0':-90, 'stride': 32, 'size':  212, 'cnt':     7}}

### VGG16 at stim size = 224 pix (not used here)
v224 = {'conv1' :{'i':  0, 'x0': -1, 'stride':  1, 'size':    3, 'cnt':   224},
        'relu1' :{'i':  1, 'x0': -1, 'stride':  1, 'size':    3, 'cnt':   224},
        'conv2' :{'i':  2, 'x0': -2, 'stride':  1, 'size':    5, 'cnt':   224},
        'relu2' :{'i':  3, 'x0': -2, 'stride':  1, 'size':    5, 'cnt':   224},
        'pool2' :{'i':  4, 'x0': -2, 'stride':  2, 'size':    6, 'cnt':   112},
        'conv3' :{'i':  5, 'x0': -4, 'stride':  2, 'size':   10, 'cnt':   112},
        'relu3' :{'i':  6, 'x0': -4, 'stride':  2, 'size':   10, 'cnt':   112},
        'conv4' :{'i':  7, 'x0': -6, 'stride':  2, 'size':   14, 'cnt':   112},
        'relu4' :{'i':  8, 'x0': -6, 'stride':  2, 'size':   14, 'cnt':   112},
        'pool4' :{'i':  9, 'x0': -6, 'stride':  4, 'size':   16, 'cnt':    56},
        'conv5' :{'i': 10, 'x0':-10, 'stride':  4, 'size':   24, 'cnt':    56},
        'relu5' :{'i': 11, 'x0':-10, 'stride':  4, 'size':   24, 'cnt':    56},
        'conv6' :{'i': 12, 'x0':-14, 'stride':  4, 'size':   32, 'cnt':    56},
        'relu6' :{'i': 13, 'x0':-14, 'stride':  4, 'size':   32, 'cnt':    56},
        'conv7' :{'i': 14, 'x0':-18, 'stride':  4, 'size':   40, 'cnt':    56},
        'relu7' :{'i': 15, 'x0':-18, 'stride':  4, 'size':   40, 'cnt':    56},
        'pool7' :{'i': 16, 'x0':-18, 'stride':  8, 'size':   44, 'cnt':    28},
        'conv8' :{'i': 17, 'x0':-26, 'stride':  8, 'size':   60, 'cnt':    28},
        'relu8' :{'i': 18, 'x0':-26, 'stride':  8, 'size':   60, 'cnt':    28},
        'conv9' :{'i': 19, 'x0':-34, 'stride':  8, 'size':   76, 'cnt':    28},
        'relu9' :{'i': 20, 'x0':-34, 'stride':  8, 'size':   76, 'cnt':    28},
        'conv10':{'i': 21, 'x0':-42, 'stride':  8, 'size':   92, 'cnt':    28},
        'relu10':{'i': 22, 'x0':-42, 'stride':  8, 'size':   92, 'cnt':    28},
        'pool10':{'i': 23, 'x0':-42, 'stride': 16, 'size':  100, 'cnt':    14},
        'conv11':{'i': 24, 'x0':-58, 'stride': 16, 'size':  132, 'cnt':    14},
        'relu11':{'i': 25, 'x0':-58, 'stride': 16, 'size':  132, 'cnt':    14},
        'conv12':{'i': 26, 'x0':-74, 'stride': 16, 'size':  164, 'cnt':    14},
        'relu12':{'i': 27, 'x0':-74, 'stride': 16, 'size':  164, 'cnt':    14},
        'conv13':{'i': 28, 'x0':-90, 'stride': 16, 'size':  196, 'cnt':    14},
        'relu13':{'i': 29, 'x0':-90, 'stride': 16, 'size':  196, 'cnt':    14},
        'pool13':{'i': 30, 'x0':-90, 'stride': 32, 'size':  212, 'cnt':     7}}




