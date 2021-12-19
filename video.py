import os
import sys
from MakeTreeDir.makedir import MAKETREEDIR

import imageio
import numpy as np

class VideoRecorder(object):
    def __init__(self, video_dir, height=512, width=512, fps=30):
        # self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        directory = MAKETREEDIR()
        directory.makedir(video_dir)
        self.save_dir = video_dir
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render("rgb_array", width=self.width, height=self.height)
            # goal_image=  env.get_goal_image(width=self.width, height=self.height)
            # final_frame=   np.concatenate((frame,goal_image), axis=1) # append across width
            assert frame is not None, "empty frame"
            # self.frames.append(final_frame)
            
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            print(path)
            imageio.mimsave(path, self.frames, fps=self.fps)