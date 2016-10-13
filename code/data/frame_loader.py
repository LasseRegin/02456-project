from __future__ import division, generators, print_function, unicode_literals, with_statement

import os
import json
import cv2
import random
import imageio

from data.persistence import DataPersistence

FILEPATH = os.path.dirname(os.path.abspath(__file__))

# It was found the using
#   video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
# to seek in the video was very slow.


class FrameFileLoader:
    DATA_FOLDER = os.path.join(FILEPATH, 'raw')

    def __init__(self, shuffle=False, **kwargs):
        self.shuffle = shuffle

        # Check data persistency
        self.data = DataPersistence(require_videos=False, require_frames=True, **kwargs)

        self.frames = []
        for video in self.data.videos:
            with open(video['annotation']) as f:
                data = json.load(f)

            balls = data['balls']
            for frame_number in range(0, video['frame_count']):
                frame_filename = os.path.join(video['frame_folder'], 'frame-%d.png' % (frame_number+1))
                ball = balls.get(str(frame_number), None)
                found = ball is not None

                if found:
                    x, y = ball['x'], ball['y']
                else:
                    x, y = None, None

                self.frames.append(Frame(
                    filename=frame_filename,
                    found=found,
                    x=x,
                    y=y
                ))

    def __iter__(self):
        if self.shuffle:
            order = np.random.permutation(len(self.frames))
            return iter(self.frames[order])
        else:
            return iter(self.frames)


class FrameLoader:
    def __init__(self, **kwargs):
        self.frame_file_loader = FrameFileLoader(**kwargs)

    def __iter__(self):
        for frame in self.frame_file_loader:
            frame.image = imageio.imread(uri=frame.filename)
            #frame.image = frame.image.mean(axis=2)
            del frame.filename
            yield frame


class FrameLoaderFromVideo:
    DATA_FOLDER = os.path.join(FILEPATH, 'raw')

    def __init__(self, shuffle=False, **kwargs):
        # Check data persistency
        self.shuffle = shuffle
        self.data = DataPersistence(require_videos=True, **kwargs)

    def initialize_video(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        fps           = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count   = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return video_capture, fps, frame_count

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data.videos)

        for video in self.data.videos:
            with open(video['annotation']) as f:
                data = json.load(f)
            balls = data['balls']

            # Initialize video capture
            video_capture, fps, frame_count = self.initialize_video(video_path=video['filename'])

            self.iter = 0
            while video_capture.isOpened():

                # Check next frame
                success = video_capture.grab()
                if not success: break
                self.iter += 1

                # Get frame information
                ball = balls.get(str(self.iter), None)
                found = ball is not None

                if found:
                    x, y = ball['x'], ball['y']
                else:
                    x, y = None, None

                # Decode frame
                _, img = video_capture.retrieve()

                # Convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                yield Frame(
                    image=img,
                    x=x,
                    y=y,
                    found=found
                )


class Frame:
    def __init__(self, x, y, found, filename=None, image=None):
        self.x = x
        self.y = y
        self.found = found
        self.filename = filename
        self.image = image
