
import math
import data
from matplotlib import pyplot as plt

frame_loader = data.FrameLoader(filename='GOPR2477')

plt.ion()
fig = plt.figure(figsize=(20,10))
img = None

for i, frame in enumerate(frame_loader):

    if (i % 100) == 0:
        print(i)

    if frame.found:
        if img is None:
            img = plt.imshow(frame.image)
        else:
            img.set_data(frame.image)

        x = frame.x * frame.image.shape[1]
        y = frame.y * frame.image.shape[0]
        print((x, y))
        plt.scatter(x, y, marker='o', s=50)

        plt.pause(.01)
        plt.draw()
        #if i > 20:
        #    break
