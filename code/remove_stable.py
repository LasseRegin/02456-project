
# -*- coding: utf-8 -*-

import numpy as np
from scipy import misc
import os as os


a=np.array([1,2,3,3,6,3,4,5,6,3,2,5])
b=np.array([1,2,3,3,6,3,4,5,6,3,4,5])
c=np.array([1,2,4,5,6,6,4,5,2,3,1,5])
d=np.array([2,4,5,5,3,2,1,2,4,5,6,7])

frames=np.array([a,b,c,d])


# read in data

sti="data/raw/fce202f968f21f70cd5972982aae5695/"
L=[]
im = misc.imread(sti+"/frame-1.png")
m=im.size
names = os.listdir(sti)
for i in range(1,len(names)+1):
    s=sti+"frame-"+str(i)+".png"
    im_stacked = misc.imread(s).reshape(m,1)
    L.append(im_stacked)
L=np.array(L)
print(L[2])


def remove_stable(frames,threshold):
    N=frames.shape[0]
    n=N    
    i=0
    while i<n-1:
        a=frames[i]
        b=frames[i+1]
        if np.abs(a-b).mean()<threshold:
            #print("hej",i)
            frames = np.delete(frames, i+1, 0)
            #print(frames)
            n=n-1
            i=i-1
        i=i+1
    n_removed = N-n
    return frames,n_removed
        
frame_new, n_rem = remove_stable(L,3)
#print(frame_new,n_rem)
frames=L
a=frames[0]
b=frames[1]
print(np.abs(a-b).mean())