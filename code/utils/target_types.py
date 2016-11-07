
import math
import numpy as np
from scipy.stats import multivariate_normal


class BallPositionPoint:
    """
        Creates an iterator returning the tuple: (image, point)
        Where
            -`image` is the image of the frames given as a numpy array.
            -`point` is the fraction-coordinates of the ball position
                     e.g. [0.5, 0.5] if the ball is in the middle of the image.
    """
    def __init__(self, frame_iterator):
        self.frame_iterator = frame_iterator

    def __iter__(self):
        for frame in self.frame_iterator:
            yield frame.image, np.array([frame.x, frame.y])


# TODO: Precompute these maps if they are needed (maybe pass frame idx).

class BallPositionPDF:
    """
        Creates an iterator returning the tuple: (image, pdf_map)
        Where
            -`image`    is the image of the frames given as a numpy array.
            -`pdf_map`  is a 2-d pdf of same size as `image` containing
                        the probability of the ball being in the given point.
    """
    def __init__(self, frame_iterator):
        self.frame_iterator = frame_iterator

    def __iter__(self):
        for frame in self.frame_iterator:
            pdf_map = np.zeros(frame.image.shape)

            if frame.found:
                # Determine position of ball
                height, width = frame.image.shape[0], frame.image.shape[1]
                x = int(frame.x * width)
                y = int(frame.y * height)

                # Crop image around ball position and mean over color channels
                width_fraction  = width  / 10
                height_fraction = height / 10
                window_x = np.array([x - width_fraction,  x + width_fraction ]).clip(0, width -1).astype(int)
                window_y = np.array([y - height_fraction, y + height_fraction]).clip(0, height-1).astype(int)
                img_cropped = frame.image[window_y[0]:window_y[1], window_x[0]:window_x[1]].mean(axis=2)

                # Count white pixels in gray cropped image and approximate the
                # diameter of the ball in terms of pixels.
                # (threshold have been manually chosen)
                white_count = np.sum(img_cropped.reshape((-1)) > 150)
                r = np.sqrt(white_count / math.pi)
                d = max(r * 2, (width_fraction + height_fraction) / 4.0)

                # Construct multivariate normal distribution for the ball position
                mesh_x, mesh_y = np.meshgrid(np.arange(0, width), np.arange(0, height))
                pos = np.empty(mesh_x.shape + (2,))
                pos[:, :, 0] = mesh_x
                pos[:, :, 1] = mesh_y
                pdf_map = multivariate_normal.pdf(x=pos, mean=np.array([x, y]), cov=np.identity(2) * d)

            yield frame.image, pdf_map


def my_norm(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


def target_output(is_ball,b,x_max,y_max): # specify b=(y,x)
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

    return output
