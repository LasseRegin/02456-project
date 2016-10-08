
import os
import json
import cv2
from scipy.misc import imread

FILEPATH = os.path.dirname(os.path.abspath(__file__))

# It was found the using
#   video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
# to seek in the video was very slow.

class FrameLoader:
    DATA_FOLDER = os.path.join(FILEPATH, 'raw')
    downsample_ext = ''

    def __init__(self, filename='GOPR2471', downsample=1, found_only=False):
        self.filename = filename
        self.downsample = downsample
        self.found_only = found_only

        # Determine downsample extension
        if self.downsample > 1:
            self.downsample_ext = '_ds_x%d' % (self.downsample)

        # Define video path
        self.VIDEO_PATH = os.path.join(
            self.DATA_FOLDER,
            '%s%s.mp4' % (self.filename, self.downsample_ext)
        )

        # Define json file path
        self.JSON_PATH = os.path.join(
            self.DATA_FOLDER,
            '%s.json' % (self.filename)
        )

        with open(self.JSON_PATH) as f:
            self.frame_data = json.load(f)

    def initialize_video(self):
        self.video_capture = cv2.VideoCapture(self.VIDEO_PATH)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    def __iter__(self):
        self.iter = 0

        self.initialize_video()

        while self.video_capture.isOpened():

            # Check next frame
            success = self.video_capture.grab()
            if not success: raise StopIteration()
            self.iter += 1

            # Get frame information
            info = self.frame_data[self.iter]

            if self.found_only and not info['found']:   continue

            # Decode frame
            _, img = self.video_capture.retrieve()

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            yield Frame(
                image=img,
                x=info['pixel_x'],
                y=info['pixel_y'],
                found=info['found']
            )


class FrameLoaderFromFiles(FrameLoader):
    def __init__(self, frame_ext='png', **kwargs):
        super().__init__(**kwargs)

        self.frame_ext = frame_ext

        # Get frame folder path
        self.FRAME_FOLDER = os.path.join(
            self.DATA_FOLDER,
            '%s%s' % (self.filename, self.downsample_ext)
        )

    def get_frame_filepath(self, frame_number):
        return os.path.join(
            self.FRAME_FOLDER,
            'frame-%d.%s' % (frame_number+1, self.frame_ext)
        )

    def __iter__(self):
        for info in self.frame_data:
            found = info['found']

            if self.found_only and not found:   continue

            img = imread(name=self.get_frame_filepath(frame_number=info['frame']))
            #img = cv2.imread(self.get_frame_filepath(frame_number=info['frame']))

            yield Frame(
                image=img,
                x=info['pixel_x'],
                y=info['pixel_y'],
                found=found
            )


class Frame:
    def __init__(self, image, x, y, found):
        self.image = image
        self.x = x
        self.y = y
        self.found = found


if __name__ == '__main__':

    import time

    start = time.time()
    frame_loader = FrameLoader(downsample=4)
    for frame in frame_loader:
        tmp = frame
    print('FrameLoader spent %.4fs' % (time.time() - start))

    start = time.time()
    frame_loader = FrameLoaderFromFiles(downsample=4)
    for frame in frame_loader:
        tmp = frame
    print('FrameLoader spent %.4fs' % (time.time() - start))
