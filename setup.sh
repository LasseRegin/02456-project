
# Get base directory
BASEDIR=$(dirname "$0")

# # Settings (original fps=30)
# FRAME_RATE=30

# Create data folder
DATA_FOLDER="$BASEDIR/code/data/raw"
FRAMES_FOLDER="$DATA_FOLDER/GOPR2471_ds_x4"
mkdir -p -- "$DATA_FOLDER"
mkdir -p -- "$FRAMES_FOLDER"

# Define video path
FILENAME="GOPR2471"
VIDEO_PATH="$DATA_FOLDER/$FILENAME.mp4"
JSON_PATH="$DATA_FOLDER/$FILENAME.json"
TMP_VIDEO_PATH="$DATA_FOLDER/$FILENAME_tmp.mp4"

# Download video sample
echo "Downloading video $FILENAME"
curl "http://recoordio-zoo.s3-eu-west-1.amazonaws.com/dataset/$FILENAME.json" > $JSON_PATH
curl "http://recoordio-zoo.s3-eu-west-1.amazonaws.com/2016/Hero4_RIG/25062016/L/GP010161.MP4" > $VIDEO_PATH

# Convert video in order to be able to extract frames using OpenCV
ffmpeg -i $VIDEO_PATH -crf 18 $TMP_VIDEO_PATH

# Also extract frames from video if this method should be used
ffmpeg -i $VIDEO_PATH scale=960:540 "$FRAMES_FOLDER/frame-%d.png"

# Clean up
rm -f $VIDEO_PATH
mv $TMP_VIDEO_PATH $VIDEO_PATH
