# 02456-project
Project in course 02456 Deep Learning E16 regarding end-to-end object tracking in high-resolution videos.

# Requirements

* Python 3
* ffmpeg
* OpenCV for Python 3 (install on [OS](http://www.pyimagesearch.com/2015/06/29/install-opencv-3-0-and-python-3-4-on-osx/))

# Setup

* Run `setup.sh` which does the following
  * Downloads one of the available videos
  * Download corresponding frame details in `JSON` format
  * Converts video using `ffmpeg` in order to make us of OpenCV's `VideoCapture` class
  * Extracts frames from the video using `ffmpeg` downsampled 4 times
  (using full size will take up about 120GB of your hard drive)
