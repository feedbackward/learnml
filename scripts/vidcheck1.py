
import numpy as np
import tables
import time
import matplotlib
from matplotlib import pyplot as plt

# Open file connection.
f = tables.open_file("../data/vim-2/Stimuli.mat", mode="r")

# Get object and array.
stimulus_object = f.get_node(where="/", name="sv")
stimulus_array = stimulus_object.read()

# Iterate over the time dimension to "play a video".
num_frames = stimulus_array.shape[0]
num_channels = stimulus_array.shape[1]
frame_w = stimulus_array.shape[2]
frame_h = stimulus_array.shape[3]

frames_to_play = 500

oneframe = np.zeros(num_channels*frame_h*frame_w, dtype=np.uint8).reshape((frame_h, frame_w, num_channels))
im = plt.imshow(oneframe)

for t in range(frames_to_play): # can make the full video if desired.
    oneframe[:,:,0] = stimulus_array[t,0,:,:] # red
    oneframe[:,:,1] = stimulus_array[t,1,:,:] # green
    oneframe[:,:,2] = stimulus_array[t,2,:,:] # blue
    im.set_data(oneframe)
    plt.pause(0.025)
plt.show()

# Close file connection.
f.close()
if not f.isopen:
    print("File connection closed.")
else:
    print("The file is still open?")
