

class BallFoundFilter:
    def __init__(self, frame_iterator):
        self.frame_iterator = frame_iterator

    def __iter__(self):
        for frame in self.frame_iterator:
            if frame.found:
                yield frame
