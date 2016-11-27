
import os
import math
import json

import data



MAX_VIDEOS = math.inf
if 'RUNNING_ON_LOCAL' in os.environ:
    MAX_VIDEOS = 4

data = data.DataPersistence(max_videos=MAX_VIDEOS)

ball_sightings = 0
counts = 0
for video in data.videos:
    with open(video['annotation'], 'r') as f:
        annotation = json.load(f)

    balls = annotation['balls']

    for i in range(0, video['frame_count']):
        # Get ball info
        ball = balls.get(str(i), None)
        found = ball is not None

        if found:
            ball_sightings += 1
        counts += 1

print('Ball sightings: %d/%d (%.4f%%)' % (ball_sightings, counts, 100 * ball_sightings / counts))
