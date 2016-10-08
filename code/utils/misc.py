

class ReshapeAndStandardize:
    def __init__(self, frame_iterator):
        self.frame_iterator = frame_iterator

    def __iter__(self):
        for image, target in self.frame_iterator:

            # Reshape image to follow the shape standards
            # (observations, channels, height, width)
            image = image.reshape(((1,) + image.shape[0:2]))

            # Standardize image
            #image = (image - image.mean()) / image.std()
            image = (image - image.mean())

            yield image.astype('float32'), target.astype('float32')
