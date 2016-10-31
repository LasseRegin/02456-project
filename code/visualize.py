
import data
from matplotlib import pyplot as plt

frame_loader = data.FrameLoader()

for i, (image, target) in enumerate(frame_loader):
    if i < 5500:    continue

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    print('Pixel coordinates')
    print('target')
    print(target)
    print(image.shape)
    axes[0].imshow(image)
    axes[1].imshow(target[0:-1].reshape((12, 20)))

    plt.show()
    if i > 5520:    break
