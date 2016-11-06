
import numpy as np
import matplotlib.pylab as plt

def my_norm(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def my_output(is_ball,b,x_max,y_max):
    output = np.zeros((x_max*y_max+1,1))
    if not is_ball or b==(-1,-1):
        output[-1]=1
        return output        
    
    mat = np.zeros((y_max,x_max))
    for i in range(y_max):
        for j in range(x_max):
            if np.abs(b[0]-i)<=1 and np.abs(b[1]-j)<=1:
                if b!=(i,j):
                    mat[i,j] = 1/my_norm(b,(i,j))/2
                else: mat[i,j]=1
    
    output[0:x_max*y_max]=mat.reshape((x_max*y_max,1))
    return output
    
# Testing 4 corner points,  point inside,  no ball,
# point pos outside and point neg outside
n = 33
b = np.array([[0,n-1,n-1,0,n//2,-1, n*2, -n], # (y,x)
             [0,0,n-1,n-1,n//2,-1, n*2, -n]])
for i in range(b.shape[1]):
    ball = (b[1,i],b[0,i]) # (y,x)
    print("(y,x)={}".format(ball))
    L = my_output(True,ball,n,n)
    print("total sum in L: {}".format(L.sum()))
    mat = L[:-1].reshape((n,n))
    print([ball[0],ball[0]+2],[ball[1],ball[1]+2])
    print(mat[ball[0]:ball[0]+2,ball[1]:ball[1]+2])
    plt.imshow(mat)
    plt.colorbar()
    plt.show()