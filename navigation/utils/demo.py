import time
import os
import pickle as pkl
from pathlib import Path
import gzip
import pickletools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from navigation.utils.image_processing import process_realsense
from navigation.commandnet.commandNN import CommandNet

class Demo:

    def __init__(self,log_root , extract=False, use_nn = False, save_video = False, use_rgb_viewing=False, run_num = None):

        # log attributes
        self.log_root = log_root
        self.log_filename=''
        self.log_count=1
        self.log_iter_count=1

        # viewing demo
        self.use_nn = use_nn
        self.save_video = save_video
        self.use_rgb_viewing = use_rgb_viewing
        self.demos = {}
        self.run_num = run_num
        self.log_path = f'{self.log_root}run{run_num}'
        self.extract = extract

        if self.use_nn:
            self.model = CommandNet(model_name='resnet18', demo_folder='simple', scaled_commands=False)
            self.model.load_trained()
        
        elif self.extract:
            self.extract_from_robot()
            self.update_demos()

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        # load old demos
        self.old_demos = None

        # new demo
        self.log = {}
        self._reset_log()

        self.prepare_flag=False

        # logging bool
        self.logging = False


    def init_log(self):

        # avoids multiple calls in sim when button pressed is read multiple times
        if self.prepare_flag:
            return
        
        # reset log count
        self.log_count = 1

        # reset log iters
        self.log_iter_count = 1

        # make base log filenname
        num_runs = len(os.listdir(self.log_root))
        curr_run = num_runs+1
        os.makedirs(self.log_root+f'/run{curr_run}')
        self.log_filename = self.log_root+f'/run{curr_run}/'

        print(f'{num_runs} previous logs')

        # turn logging on
        self.logging = True

        # start log time
        self.log_start_time = time.time()

        self.prepare_flag=True

        print('Starting log...')

    def save_partial_log(self):
        # make partial log filename
        log_name = self.log_filename+f'log{self.log_count}.pkl'

        # update log count
        self.log_count+=1

        self.log['Time']=time.time()-self.log_start_time

        # update log start time
        self.log_start_time = time.time()

        # save log
        print('Saving partial log...')
        with open(log_name, 'wb') as file:
            pkl.dump(self.log, file, pkl.HIGHEST_PROTOCOL)  

        print('Partial log saved to', log_name)

        # reset log
        self._reset_log()

    def end_log(self, no_save=False):
        # set logging to false
        self.logging = False

        # save last log
        if not no_save:

            self.save_partial_log()

        else:
            self.log = self._reset_log()

        print('Log ended.')



    def collect_demo_data(self, data):
        # data should be dict

        for k in data:
            self.log[k].append(data[k])

        self.log_iter_count+=1


        print('Collected data count:', len(self.log['Commands']))

    
    def extract_from_robot(self):
        print('Extracting from robot')
        os.system(f'rsync -a -P --info=progress2 --ignore-existing  unitree@192.168.123.15:go1_gym/logs/jenkins_experiment/ navigation/robot_demos/jenkins_experiment/{self.demo_folder}/runs/')

    
    def extract_demos(self):
        print('Extracting data')

      
        log_files = sorted([str(p) for p in Path(self.log_path).glob("*.pkl")],key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))
        for log in log_files:
            with gzip.open(log, 'rb') as f:
                p = pkl.Unpickler(f)
                demo = p.load()

            for k in demo:
                if k=='Time':
                    self.log[k]+=demo[k]
                else:
                    print(len(demo[k]))
                    self.log[k]+=demo[k]

        rgb=self.log['Image1st']
        #self.depth=self.demos['DepthImg']
        commands=self.log['Commands']

        rounded_comms = [[round(com,3) for com in c] for c in commands]

        print('Time of demo:', self.demos['Time'])

        return rgb, rounded_comms

    def view_single(self,indx=None):
        
        fig, ax = plt.subplots(1,1)
        if not indx:
            indx = random.randint(0,len(self.rgb)-1)
          
        ax[0].imshow(self.rgb[indx])
        ax[0].set_title('RGB')
        # ax[1].imshow(self.depth[indx])
        # ax[1].set_title('Depth')
        fig.suptitle(f'Command: {self.rounded_comms[indx]}')
        plt.show()


    def view_video(self):
        from torchvision import transforms
        to_pil=transforms.Compose([
        transforms.ToPILImage()]
    )
        rgb, rounded_comms = self.extract_demos()

        print('Preparing video...')
        fig,ax=plt.subplots(1,2)
        fig.set_size_inches(20,10)
        fig.suptitle(f'Run{self.run_num}')


        ax[0].set_title('RGB')
        #ax[1].set_title('Depth')
        if self.use_nn:
            ax[1].text(0.01, 0.5, 'Pred Commands: \n True Commands: ')
        else:
            ax[1].text(0.01, 0.5, 'Commands: ')

        movie=[]
        for f in range(0,len(self.rgb)):

            img_rgb = rgb[f]
            img= process_realsense(img_rgb, deploy=True)[0]

            
            if self.use_nn:
                commands, policy = self.model(img)
                commands, policy = self.model._data_rescale(commands, policy)

                commands.append(policy)
                commands = [round(c,4) for c in commands]

                movie.append([ax[0].imshow(to_pil(img),animated=True),ax[1].text(0.01, 0.5, f'Pred Commands: {commands}\n True Commands: {rounded_comms[f]}')])


            else:

                if self.use_rgb_viewing:
                    show_img=self.rgb[f]
                else:
                    show_img=to_pil(img)
                movie.append([ax[0].imshow(show_img,animated=True),ax[1].text(0.01, 0.5, f'Commands: {rounded_comms[f]}')])
        
        ani = animation.ArtistAnimation(fig,movie,interval=100,blit=True)

        if self.save_video:
            ani.save(f'video_run{self.run_num}.mp4',fps=15)
            print('Video saved.')
        plt.show()

    
    def update_demos(self):
        print('Updating demos...demo',self.run_num)
        log_files = sorted([str(p) for p in Path(self.log_path).glob("*.pkl")],key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))

        
        for log in log_files:
            with open(log, 'rb') as f:
                demo = pkl.load(f)

            with gzip.open(log, "wb") as f:
                pickled = pkl.dumps(demo)
                optimized_pickle = pickletools.optimize(pickled)
                f.write(optimized_pickle)


    def _reset_log(self):
        self.log = {'Commands':[], 'Image1st':[], 'DepthImg':[], 'Time':[],'Torque':[], 'Joint_Vel':[]}



if __name__ == "__main__":

    demo_folder = 'simple'
    log_root = f'navigation/robot_demos/jenkins_experiment/{demo_folder}/runs/'
    
    extract = True
    save_video = False

    use_nn = False
    use_rgb = False

    runs = [1]
    
    for run in runs:
        demo = Demo(log_root=log_root, run_num=run, extract = extract, save_video=save_video, use_nn=use_nn, use_rgb_viewing=use_rgb)
