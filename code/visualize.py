
import data
from matplotlib import pyplot as plt

frame_loader = data.FrameLoader(shuffle=True)

for i, (image, target) in enumerate(frame_loader):
    fig = plt.figure(figsize=(20,10))
    print('Pixel coordinates')
    print(target[0]*image.shape[1],target[1]*image.shape[0])
    plt.imshow(image, cmap='gray')

    plt.show()
    if i > 20:
        break
