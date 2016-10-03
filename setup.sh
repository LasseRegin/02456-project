
# Get base directory
BASEDIR=$(dirname "$0")

# Settings (original fps=30)
FRAME_RATE=30

# Create data folder
DATA_FOLDER="$BASEDIR/code/data/raw"
FRAMES_FOLDER="$DATA_FOLDER/frames"
mkdir -p -- "$DATA_FOLDER"
mkdir -p -- "$FRAMES_FOLDER"

# Define video path
VIDEO_PATH="$DATA_FOLDER/GOPR2477.mp4"
JSON_PATH="$DATA_FOLDER/GOPR2477.json"
TMP_VIDEO_PATH="$DATA_FOLDER/GOPR2477_tmp.mp4"

# Download video sample
echo "Downloading video 2477"
curl "http://dendron.sportcaster.dk/api/video/2477/humandetection/" > $JSON_PATH
curl "http://recoordio-zoo.s3-eu-west-1.amazonaws.com/2016/Hero4_RIG/28072016/L/GOPR0187.MP4" > $VIDEO_PATH

# Convert video in order to be able to extract frames using OpenCV
ffmpeg -i $VIDEO_PATH -crf 18 $TMP_VIDEO_PATH

# Clean up
rm -f $VIDEO_PATH
mv $TMP_VIDEO_PATH $VIDEO_PATH

# Extract frames from video
#ffmpeg -i $VIDEO_PATH -vf scale=iw*0.25:ih*0.25 -r $FRAME_RATE "$FRAMES_FOLDER/frame-%d.png"
#ffmpeg -i $VIDEO_PATH -vf scale=960:540 -r $FRAME_RATE "$FRAMES_FOLDER/frame-%d.png"

# Remove movie again
#rm -f $VIDEO_PATH
