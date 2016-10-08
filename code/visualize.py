
import math
import numpy as np

import data
import utils
from matplotlib import pyplot as plt

frame_loader = data.FrameLoader(filename='GOPR2477', downsample=2)
frame_loader = utils.BallFoundFilter(frame_iterator=frame_loader)
frame_loader = utils.BallPositionPDF(frame_iterator=frame_loader)

for i, (image, pdf) in enumerate(frame_loader):
    fig = plt.figure(figsize=(20,10))
    plt.subplot(2, 1, 1)
    plt.imshow(pdf)
    plt.subplot(2, 1, 2)
    plt.imshow(image)

    plt.show()

    if i > 20:
        break
