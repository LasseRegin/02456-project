
# Ideas

## Add "action map" to input

Create an "action map"/"change map" to the input as an extra layer. This map
can contain the pixels obtained from subtracting current frame with previous
frame(s).



# Problems

When using the `VideoCapture` we cannot shuffle the frames and thereby the order
of the frames will be the same on each epoch. The only *shuffling* we can do is
shuffling the order of the frame in the minibatches.



## Other

Cool MeanShift algorithm for finding areas of interest (a ball maybe):
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_meanshift/py_meanshift.html#meanshift


Optical flow using cv2
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html#lucas-kanade


Background subtraction using cv2
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#py-background-subtraction
