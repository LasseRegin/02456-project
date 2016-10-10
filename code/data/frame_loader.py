
import os
import json
import cv2
import subprocess
from scipy.misc import imread
from urllib.request import urlopen, urlretrieve

FILEPATH = os.path.dirname(os.path.abspath(__file__))

# It was found the using
#   video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
# to seek in the video was very slow.


# TODO: Maybe fix small loading time from fetching info .json file, when we already
#       have files.
class DataPersistence:
    DATA_FOLDER = os.path.join(FILEPATH, 'raw')
    def __init__(self, info_url='http://recoordio-zoo.s3-eu-west-1.amazonaws.com/dataset/09102016.json',
                 download_missing=True, extract_frames=True):
        self.videos = []

        # Make sure raw folder exists
        self.make_folder(foldername=self.DATA_FOLDER)

        # Download data
        response = urlopen(info_url)
        data_raw = response.read().decode("utf-8")
        data = json.loads(data_raw)

        if 'videos' not in data:
            raise KeyError('Could not find videos key in json file.')

        for video in data['videos']:
            for sample in video['samples']:
                url = sample['s3video']
                basename = os.path.basename(url).split('.')[0]
                filename = os.path.join(self.DATA_FOLDER, '%s.mp4' % (basename))

                # Make frames folder
                frame_folder = os.path.join(self.DATA_FOLDER, basename)
                has_frames = self.make_folder(foldername=frame_folder)

                if not os.path.isfile(filename) and download_missing:
                    print('Downloading %s..' % (basename))
                    urlretrieve(url, filename)

                # Get annotations if video contains annotations
                if 'human_annotation' in sample:
                    url_anno = sample['human_annotation']['s3annotation']
                    filename_anno = os.path.join(self.DATA_FOLDER, os.path.basename(url_anno))

                    if not os.path.isfile(filename_anno):
                        urlretrieve(url_anno, filename=filename_anno)

                    self.videos.append({
                        'annotation': filename_anno,
                        'basename': basename
                    })


                if not has_frames and extract_frames:
                    # Extract frames
                    self.extract_frames(
                        video_path=filename,
                        target_width=480, target_height=270,
                        target_framerate=10,
                        target_filenames_formatted='%s/frame-%%d.png' % (frame_folder)
                    )

    def make_folder(self, foldername):
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
            return False
        return True

    def extract_frames(self, video_path, target_width, target_height,
                       target_framerate, target_filenames_formatted):
        # Make sure we are in the data folder
        os.chdir(FILEPATH)

        # Run command
        subprocess.call(['ffmpeg',
            '-i', video_path,
            '-s', '%dx%d' % (target_width, target_height),
            '-r', str(target_framerate),
            target_filenames_formatted
        ])



class FrameFileLoader:
    DATA_FOLDER = os.path.join(FILEPATH, 'raw')

    def __init__(self, shuffle=False, **kwargs):
        self.shuffle = shuffle

        # Check data persistency
        self.data = DataPersistence(**kwargs)


        for video in self.data.videos:
            print(video)

        return
        # Load all filenames
        self.filenames = [filename for filename in self.data.videos]

    def __iter__(self):
        if self.shuffle:
            order = np.random.permutation(len(self.filename))
            return self.filenames[order]
        else:
            return self.filenames


class FrameLoader:
    def __init__(self, **kwargs):
        self.frame_file_loader(**kwargs)

    def __iter__(self):
        for filename in self.frame_file_loader:
            print(filename)
            #img = imread(name=filename)


# class FrameLoaderFromVideo:
#     DATA_FOLDER = os.path.join(FILEPATH, 'raw')
#     downsample_ext = ''
#
#     def __init__(self, filename='GOPR2471', downsample=1, found_only=False):
#         self.filename = filename
#         self.downsample = downsample
#         self.found_only = found_only
#
#         # Check data persistency
#         self.data = DataPersistence()
#         self.data.download()
#
#         return
#
#         # Determine downsample extension
#         if self.downsample > 1:
#             self.downsample_ext = '_ds_x%d' % (self.downsample)
#
#         # Define video path
#         self.VIDEO_PATH = os.path.join(
#             self.DATA_FOLDER,
#             '%s%s.mp4' % (self.filename, self.downsample_ext)
#         )
#
#         # Define json file path
#         self.JSON_PATH = os.path.join(
#             self.DATA_FOLDER,
#             '%s.json' % (self.filename)
#         )
#
#         with open(self.JSON_PATH) as f:
#             self.frame_data = json.load(f)
#
#     def initialize_video(self):
#         self.video_capture = cv2.VideoCapture(self.VIDEO_PATH)
#         self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
#         self.frame_count = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
#
#     def __iter__(self):
#         self.iter = 0
#
#         self.initialize_video()
#
#         while self.video_capture.isOpened():
#
#             # Check next frame
#             success = self.video_capture.grab()
#             if not success: raise StopIteration()
#             self.iter += 1
#
#             # Get frame information
#             info = self.frame_data[self.iter]
#
#             if self.found_only and not info['found']:   continue
#
#             # Decode frame
#             _, img = self.video_capture.retrieve()
#
#             # Convert to grayscale
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#             yield Frame(
#                 image=img,
#                 x=info['pixel_x'],
#                 y=info['pixel_y'],
#                 found=info['found']
#             )


# class FrameLoaderFromFiles(FrameLoader):
#     def __init__(self, frame_ext='png', **kwargs):
#         super().__init__(**kwargs)
#
#         self.frame_ext = frame_ext
#
#         # Get frame folder path
#         self.FRAME_FOLDER = os.path.join(
#             self.DATA_FOLDER,
#             '%s%s' % (self.filename, self.downsample_ext)
#         )
#
#     def get_frame_filepath(self, frame_number):
#         return os.path.join(
#             self.FRAME_FOLDER,
#             'frame-%d.%s' % (frame_number+1, self.frame_ext)
#         )
#
#     def __iter__(self):
#         for info in self.frame_data:
#             found = info['found']
#
#             if self.found_only and not found:   continue
#
#             img = imread(name=self.get_frame_filepath(frame_number=info['frame']))
#             #img = cv2.imread(self.get_frame_filepath(frame_number=info['frame']))
#
#             yield Frame(
#                 image=img,
#                 x=info['pixel_x'],
#                 y=info['pixel_y'],
#                 found=found
#             )


class Frame:
    def __init__(self, image, x, y, found):
        self.image = image
        self.x = x
        self.y = y
        self.found = found


if __name__ == '__main__':

    #data_loader = FrameLoader(download_missing=False)
    frame_file_loader = FrameFileLoader(download_missing=False)

    #for frame_file in frame_file_loader:
    #    print(frame_file)

    # import time
    #
    # start = time.time()
    # frame_loader = FrameLoader(downsample=4)
    # for frame in frame_loader:
    #     tmp = frame
    # print('FrameLoader spent %.4fs' % (time.time() - start))
    #
    # start = time.time()
    # frame_loader = FrameLoaderFromFiles(downsample=4)
    # for frame in frame_loader:
    #     tmp = frame
    # print('FrameLoader spent %.4fs' % (time.time() - start))
