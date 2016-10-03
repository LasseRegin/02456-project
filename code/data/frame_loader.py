
import os
import json
import cv2
import matplotlib.pyplot as plt

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class FrameLoader:
    DATA_FOLDER = os.path.join(FILEPATH, 'raw')
    FRAMES_FOLDER = os.path.join(DATA_FOLDER, 'frames')

    def __init__(self, filename='GOPR2477'):
        self.filename = filename

        self.video_capture = cv2.VideoCapture(os.path.join(self.DATA_FOLDER, '%s.mp4' % (self.filename)))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        with open(os.path.join(self.DATA_FOLDER, '%s.json' % (self.filename))) as f:
            self.frame_data = json.load(f)

    def __iter__(self):
        self.iter = 0

        while self.video_capture.isOpened():
            success, img = self.video_capture.read()

            if not success: raise StopIteration()
            self.iter += 1
            info = self.frame_data[self.iter]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            yield Frame(
                frame_id=info['id'],
                image=img,
                x=info['pixel_x'],
                y=info['pixel_y'],
                found=info['found']
            )


class FoundFrameLoader(FrameLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        self.iter = 0

        for i, info in enumerate(self.frame_data):
            if info['found']:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, info['frame'])
                success, img = self.video_capture.read()

                if not success: raise StopIteration()

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                yield Frame(
                    frame_id=info['id'],
                    image=img,
                    x=info['pixel_x'],
                    y=info['pixel_y'],
                    found=info['found']
                )


class Frame:
    def __init__(self, frame_id, image, x, y, found):
        self.frame_id = frame_id
        self.image = image
        self.x = x
        self.y = y
        self.found = found


if __name__ == '__main__':
    frame_loader = FrameLoader()

    for i, frame in enumerate(frame_loader):
        print(i)
        print(frame.image.shape)
