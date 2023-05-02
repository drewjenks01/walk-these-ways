import imageio

class VideoRecorder:

    def __init__(self):

        self.video_imgs = []
        self.filepath = 'navigation/play_video.mp4'


    def record_img(self, env):
        self.video_imgs.append(env.render(mode='rgb_array'))

    def save_video(self):
        with imageio.get_writer(self.filepath, mode='I',fps=35) as writer:
            for img in self.video_imgs:
                writer.append_data(img)

