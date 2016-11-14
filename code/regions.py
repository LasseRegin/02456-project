import numpy as np
import data
import imageio as imo
from scipy import misc


n_frames=1500
frame_loader = data.FrameLoader(max_videos=4)
writer = imo.get_writer('predict_ball.mp4', fps=30)
for i , (image, target) in enumerate(frame_loader):    
    #print("iteration {} of {}".format(i,n_frames))
    
    #print("shape", image.shape)
    cells_y, cells_x = frame_loader.cells_y, frame_loader.cells_x
    image -= np.min(image)
    image = misc.imresize(image, size = (304,304,3))
    
    (y_max,x_max) = image.shape[0:2]
    patch_y, patch_x = y_max//cells_y, x_max//cells_x
    distribution = target[:-1].reshape((cells_y,cells_x))
    
    image = image.astype("float32")    
    
    if i==n_frames:
        break
    if np.argmax(target)!=len(target)-1:
        im_patch = np.zeros((y_max, x_max))
        for i in range(cells_y):
            for j in range(cells_x):
                im_patch[i*patch_y:(i+1)*patch_y, j*patch_x:(j+1)*patch_x] = distribution[i,j]
        val = np.max(im_patch)
        im_patch = im_patch/val*50
        #image = image.astype("float32")

        image[:,:,0] += im_patch
        if np.max(image[:,:,0])>255:
            val=np.max(image[:,:,0])
            image[:,:,0] = image[:,:,0]/val*255
        #print("min = {}, max = {}".format(np.min(image),np.max(image)))
    image = image.astype("uint8")
    writer.append_data(image)

writer.close()
