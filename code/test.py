

import data
import utils


frame_loader = data.FrameLoader()

# heatmap_output(is_ball,b,x_max,y_max)

for frame, target in data.FrameLoader():
    found = bool(target[2])
    heatmap = utils.ballPositionHeatMap(found=found, x=target[0], y=target[1], cells_x=10, cells_y=6)
    print(heatmap)
    #break
