from __future__ import division, generators, print_function, unicode_literals, with_statement

import os
import json
import subprocess
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
        `dim_ds_rate`   Indicates the dimensionality downsample rate used when
                        extracting frames using `ffmpeg`. E.g. `dim_ds_rate`=2
                        will halve the width and height.
    """
    ORIGINAL_WIDTH  = 3840
    ORIGINAL_HEIGHT = 2160

    DATA_FOLDER = os.path.join(FILEPATH, 'raw')
    def __init__(self, info_url='http://recoordio-zoo.s3-eu-west-1.amazonaws.com/dataset/09102016.json',
                 require_videos=True, require_frames=False, max_videos=None, dim_ds_rate=8):
        self.dim_ds_rate = dim_ds_rate
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
                url = sample['s3video']
                basename = os.path.basename(url)

                # Determine filename based on downsample rate
                filename = os.path.join(self.DATA_FOLDER, basename)

                # Download video file if needed
                if not os.path.isfile(filename):
                    print('Downloading %s..' % (basename))
                    urlretrieve(url, filename)

                # Create downsample video if needed and video required
                filename_ds = self.construct_ds_filename(filename)
                if self.dim_ds_rate > 1 and require_videos:
                    if not os.path.isfile(filename_ds):
                        print('Creating downsampled video')
                        self.downsample_video(
                            video_path=filename,
                            target_width=self.ORIGINAL_WIDTH   // self.dim_ds_rate,
                            target_height=self.ORIGINAL_HEIGHT // self.dim_ds_rate,
                            target_framerate=sample['fps_nominator'] / sample['fps_denominator'],
                            target_video_path=filename_ds
                        )

                # Check frames if frames are required
                if require_frames:

                    # Make frames folder if needed
                    frame_folder = os.path.join(self.DATA_FOLDER, basename.split('.')[0])
                    self.make_folder(foldername=frame_folder)

                    # Extract frames if needed
                    if len(os.listdir(frame_folder)) < sample['frame_count']:
                        self.extract_frames(
                            video_path=filename,
                            target_width=self.ORIGINAL_WIDTH   // self.dim_ds_rate,
                            target_height=self.ORIGINAL_HEIGHT // self.dim_ds_rate,
                            target_framerate=sample['fps_nominator'] / sample['fps_denominator'],
                            target_filenames_formatted='%s/frame-%%d.png' % (frame_folder)
                        )

                # Get annotations if video sample contains annotations
                if 'human_annotation' in sample:
                    url_anno = sample['human_annotation']['s3annotation']
                    filename_anno = os.path.join(self.DATA_FOLDER, os.path.basename(url_anno))

                    if not os.path.isfile(filename_anno):
                        urlretrieve(url_anno, filename=filename_anno)

                    self.videos.append({
                        'filename': filename_ds if self.dim_ds_rate > 1 else filename,
                        'annotation': filename_anno,
                        'frame_folder': frame_folder if require_frames else None,
                        'frame_count': sample['frame_count']
                    })
                    video_count += 1

                # If the argument `max_videos` is provided we limit the number of
                # videos we fetch.
                if max_videos is not None and video_count >= max_videos:
                    break
            if max_videos is not None and video_count >= max_videos:
                break


    def construct_ds_filename(self, filename):
        filename_splitted = filename.split('.')
        filename = ''.join(filename_splitted[0:len(filename_splitted)-1])

        # Add dimensionality reduction term
        filename += '-dim-ds-x%d' % (self.dim_ds_rate)

        filename += '.%s' % (filename_splitted[-1])
        return filename

    def make_folder(self, foldername):
        if not os.path.isdir(foldername):
            os.mkdir(foldername)

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

    def downsample_video(self, video_path, target_width, target_height,
                         target_framerate, target_video_path):
        # Make sure we are in the data folder
        os.chdir(FILEPATH)

        # Run command
        subprocess.call(['ffmpeg',
            '-i', video_path,
            '-crf', '18',
            '-s', '%dx%d' % (target_width, target_height),
            '-r', str(target_framerate),
            '-an', # Remove audio track
            target_video_path
        ])

if __name__ == '__main__':
    data_persistence = DataPersistence()
