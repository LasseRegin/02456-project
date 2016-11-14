from __future__ import division, generators, print_function, unicode_literals, with_statement

import os
import json
import hashlib
import subprocess
import math
from urllib.request import urlopen, urlretrieve

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class DataPersistence:
    """
        Class used for downloading videos and video JSON files.
        When initialized the object checks if all videos and corresponding JSON
        files have been downloaded to the `DATA_FOLDER` path, and checks if
        frames have been extracted.

        `info_url`      URL to JSON file containing video + annotation information.
        `max_videos`    Restricts the number of loaded videos to `max_videos`.
    """
    ORIGINAL_WIDTH  = 3840
    ORIGINAL_HEIGHT = 2160

    DATA_FOLDER = os.path.join(FILEPATH, 'raw')
    def __init__(self, info_url='http://recoordio-zoo.s3-eu-west-1.amazonaws.com/dataset/09102016.json',
                 max_videos=math.inf, target_width=299, target_height=299):
        # Set target shapes
        self.target_width = target_width
        self.target_height = target_height
        self.videos = []

        # Make sure raw folder exists
        self.make_folder(foldername=self.DATA_FOLDER)

        # Get video information
        info_filename = os.path.join(self.DATA_FOLDER, 'video-info.json')
        if not os.path.isfile(info_filename):
            response = urlopen(info_url)
            data_raw = response.read().decode("utf-8")
            with open(info_filename, 'w') as f:
                f.write(data_raw)

        with open(info_filename, 'r') as f:
            data = json.load(f)

        if 'videos' not in data:
            raise KeyError('Could not find \"videos\" key in json file.')

        # Sort videos to make sure we get the same order every time
        video_count = 0
        for video in sorted(data['videos'], key=lambda vid: vid['hash']):
            for sample in video['samples']:
                # If video sample has no annotation just continue
                if not 'human_annotation' in sample:    continue

                url = sample['s3video']
                basename = os.path.basename(url)

                # Create filename
                filename = os.path.join(self.DATA_FOLDER, basename)

                # Download video file if needed
                if not os.path.isfile(filename):
                    print('Downloading %s..' % (basename))
                    urlretrieve(url, filename)

                # Create downsample video if needed and video required
                downsample = self.target_width != self.ORIGINAL_WIDTH or self.target_height != self.ORIGINAL_HEIGHT
                foldername = self.construct_ds_foldername(filename)
                if downsample:
                    self.make_folder(foldername=foldername)
                    frame_filenames = self.get_frames_from_folder(foldername=foldername)

                    if len(frame_filenames) != sample['frame_count']:
                        self.extract_frames(
                            video_path=filename,
                            target_width=self.target_width,
                            target_height=self.target_height,
                            folder_path=foldername
                        )
                    #else:
                    #    # Delete video file again
                    #    os.remove(filename)

                # Get video annotations
                url_anno = sample['human_annotation']['s3annotation']
                filename_anno = os.path.join(self.DATA_FOLDER, os.path.basename(url_anno))

                # Download annotation file if we don't have it already
                if not os.path.isfile(filename_anno):
                    urlretrieve(url_anno, filename=filename_anno)

                self.videos.append({
                    'foldername': foldername,
                    'annotation': filename_anno,
                    'frame_count': sample['frame_count'],
                    'filename': filename
                })
                video_count += 1

                # If the argument `max_videos` is provided we limit the number of
                # videos we fetch.
                if video_count >= max_videos:
                    break
            if video_count >= max_videos:
                break

    def construct_ds_foldername(self, filename):
        filename_splitted = filename.split('.')
        foldername = ''.join(filename_splitted[0:-1])

        # Add dimensionality term
        foldername += '-dim-%d-x-%d' % (self.target_width, self.target_height)

        return foldername

    def make_folder(self, foldername):
        if not os.path.isdir(foldername):
            os.mkdir(foldername)

    def extract_frames(self, video_path, target_width, target_height, folder_path):
        print('Extracting frames to %s' % (folder_path))

        # Make sure we are in the data folder
        os.chdir(FILEPATH)

        # Make dir for frames
        #target_video_path = '%s/%%d.bmp' % (folder_path)
        target_video_path = '%s/%%d.png' % (folder_path)

        # Run command
        subprocess.call(['ffmpeg',
            '-i', video_path,
            '-s', '%dx%d' % (target_width, target_height),
            target_video_path
        ])

    def get_frames_from_folder(self, foldername):
        return list(filter(lambda name: not name.startswith('.'), os.listdir(foldername)))

    def __hash__(self):
        m = hashlib.md5()
        joined_str = ''.join(sorted([v['foldername'] for v in self.videos])).encode('utf-8')
        m.update(joined_str)
        return int(m.hexdigest(), 16)

if __name__ == '__main__':
    data_persistence = DataPersistence()
