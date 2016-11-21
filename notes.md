# Presentation
Presentation:
* introduktion af problemstilling
* data (billede af video, statistik fra data)
* preprocessering af data (trækker mean fra, target map)
* logistic regression 
* cnn (inception graf)
* resultater (loss curve, summary statistik, farve boldområde)
* demo (bedste model)

# TODOs

Lasse
* Kør frameloader på dendron for at tælle hvor mange frames der er annoterede
* Find best model (be the best you can be)
* summary statitics
* vælg god testvideo og udeluk fra data

Nikolaj
* Convert video script to work on full HD (and Lasse)
* Find out basic statistics about the dataset
* Begynd på slides


# Meetings

## 10/10-2016:

Notes from the meeting

* The precision of the ball doesn't have to be very exact
* Idea about adding motiontracking as an extra channel to the input
* We should use their newest dataset
* Maybe use pretraining Tensorflow models
* Downsample frames to arround 512x512 and fps to only use every 3rd frame.
* Expand our frame loader to support multiple videos and maybe support URL

For next week

* Upload our frame extracting script to their repository
* Fix preprocessing
* Their email addresses are:
  * philiphenningsen@gmail.com
  * jesper@sportcaster.dk
* On their server we should use a virtual environment e.g. Anaconda virtualenv
* Maybe try ImageIO library for loading images
* Try OpenCV on the new videos


## 24/10-2016:

Questions:
* The loaded frozen graph:
  * Does it require the fixed input size of 299x299?
* Have any good way of loading the frames?
  * Can we maybe exploit the large amount of memory on the dendron machine?



Notes from the meeting

*

For next week

* Change to predicting regions with softmax (maybe use weighted)
  * Split in 20*12 regions and create targets being 0's in all regions

## 7/10-2016

For next week do:


Lasse
* Ask Philip about the loss of spatial information using the Inception graph
* Create persistency for Inception graph output
* Test with logistic regression


Nikolaj
* Finish video evaluation script



# Timings

## Loading frames

Frames of size `480x270` with a total of approximate **33,000** frames

| Method           | Loading time  |
| -------------    |:-------------:|
| Use Pickle       | 38.67s        |
| Read images\*    | 287.87        |
| Read images\*\*  | 169.97        |
| Load Video\*\*\* | 18.22         |

\* Using `scipy.misc.imread` <br>
\*\* Using `ImageIO` <br>
\*\*\* Using OpenCV

# Ideas


Use Pickle to save the frames!


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
