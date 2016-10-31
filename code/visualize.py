
import data
from matplotlib import pyplot as plt

frame_loader = data.FrameLoader()

for i, (image, target) in enumerate(frame_loader):
    #found = bool(target[2])
    #if not found:   continue
    fig = plt.figure(figsize=(20,10))
    print('Pixel coordinates')
    print('target')
    print(target[0]*image.shape[1],target[1]*image.shape[0])
    print(image.shape)
    plt.imshow(image)

    plt.show()
    if i > 20:
        break
