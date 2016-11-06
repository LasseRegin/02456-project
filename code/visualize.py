
import data
from matplotlib import pyplot as plt

#frame_loader = data.FrameLoader()
frame_loader = data.FrameLoader(max_videos=4)

for i, (image, target) in enumerate(frame_loader):
    #if i < 5500:    continue
    if i < 1000: continue

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Temp fix for getting proper value range
    image -= image.min()
    image /= image.max()
    image *= 255
    image = image.astype('uint8')

    axes[0].imshow(image)
    axes[1].imshow(target[0:-1].reshape((12, 20)))

    plt.show()
    if i > 1010: break
    #if i > 5520:    break
