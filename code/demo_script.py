import numpy as np
import data
import imageio as imo
from scipy import misc

filename_test_video = 'test_video.mp4'
reader = imo.get_reader(filename_test_video) # remember extension
fps = reader.get_meta_data()['fps']
writer = imo.get_writer('demo_test.mp4', fps=fps)
for im_hd in reader:
    image = misc.imresize(im_hd, size = (299,299,3))
    image -= np.mean(image)
    target = predict(image) #psedo kode
    
    cells_y, cells_x = 20, 12
    (y_max,x_max) = im_hd.shape[0:2]
    patch_y, patch_x = y_max//cells_y, x_max//cells_x
    distribution = target[:-1].reshape((cells_y,cells_x))
    
    im_hd = im_hd.astype("float32")
    if np.argmax(target)!=len(target)-1:
        im_patch = np.zeros((y_max, x_max))
        for i in range(cells_y):
            for j in range(cells_x):
                im_patch[i*patch_y:(i+1)*patch_y, j*patch_x:(j+1)*patch_x] = distribution[i,j]
        val = np.max(im_patch)
        im_patch = im_patch/val*50
        #im_hd = im_hd.astype("float32")

        im_hd[:,:,0] += im_patch
        if np.max(im_hd[:,:,0])>255:
            val=np.max(im_hd[:,:,0])
            im_hd[:,:,0] = im_hd[:,:,0]/val*255
        #print("min = {}, max = {}".format(np.min(im_hd),np.max(im_hd)))
    im_hd = im_hd.astype("uint8")
    writer.append_data(im_hd)

writer.close()

