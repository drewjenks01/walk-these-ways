import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import numpy as np
import random
import os
from pathlib import Path
from navigation.utils.image_processing import process_realsense, load_trained
import gzip, pickletools

class DemoViewer:

    def __init__(self, run_num, extract, save_video, demo_folder, use_nn,use_rgb):
        self.run_num=run_num
        self.extract=extract
        self.save_video=save_video
        self.demo_folder =demo_folder
        self.use_nn=use_nn
        self.use_rgb=use_rgb

        if use_nn:
            self.model = load_trained(image_mode='first', model_name='mnv3s',demo_folder='simple', scaled_commands=True)


        self.demos={}

        self.root_path = f'navigation/robot_demos/jenkins_experiment/{self.demo_folder}/runs/'

        self.log_path = f'{self.root_path}run{run_num}'

        self.demos = {'Commands':[], 'Image1st':[], 'DepthImg':[],'Time':0}

        if self.extract: 
            self.extract_from_robot()
            self.update_demos()

        self.rgb=[]
        #self.depth=self.demos['DepthImg']
        self.commands=[]

        self.rounded_comms = []
        

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
        self.extract_demos()

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

            img_rgb = self.rgb[f]
            img= process_realsense(img_rgb, deploy=True)[0]

            
            if self.use_nn:
                commands, policy = self.model(img)
                commands, policy = self.model._data_rescale(commands, policy)

                commands.append(policy)
                commands = [round(c,4) for c in commands]

                movie.append([ax[0].imshow(to_pil(img),animated=True),ax[1].text(0.01, 0.5, f'Pred Commands: {commands}\n True Commands: {self.rounded_comms[f]}')])


            else:

                if self.use_rgb:
                    show_img=self.rgb[f]
                else:
                    show_img=to_pil(img)
                movie.append([ax[0].imshow(show_img,animated=True),ax[1].text(0.01, 0.5, f'Commands: {self.rounded_comms[f]}')])
        
        ani = animation.ArtistAnimation(fig,movie,interval=100,blit=True)

        if self.save_video:
            ani.save(f'video_run{self.run_num}.mp4',fps=15)
            print('Video saved.')
        plt.show()


    def download_all_data(self, folder):
        print('Downloading ALL data...')

        num_runs = len([p for p in Path(self.root_path).glob('*')])
        print('Num runs', num_runs)
        
        comms=[]
        test_comms=[]
        images=[]
        test_images=[]

        for i in range(num_runs):
            log_files = sorted([str(p) for p in Path(self.root_path+f'run{i+1}').glob("*.pkl")],key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))
        
            for log in log_files:
                with open(log, 'rb') as f:
                    demo = pickle.load(f)
                    for k in demo:

                        if k =='Commands':
                            if i ==num_runs-1:
                                test_comms+=demo[k]
                            else:
                                comms+=demo[k]
                        elif k =='Image1st':
                            if i ==num_runs-1:
                                test_images+=demo[k]
                            else:
                                images+=demo[k]
                                

        
        comms=np.array(comms)
        images=np.array(images)
        test_comms = np.array(test_comms)
        test_images = np.array(test_images)

        print('Shapes',comms.shape, images.shape, test_comms.shape, test_images.shape)
        logs={'Commands':comms, 'Image1st':images, 'Test_Commands':test_comms, 'Test_Image1st':test_images, 'Count':num_runs}
        with open(f'navigation/robot_demos/jenkins_experiment/{folder}/demos.pkl', 'wb') as file:
            pickle.dump(logs, file)  


    def extract_from_robot(self):
        print('Extracting from robot')
        os.system(f'rsync -a -P --info=progress2 --ignore-existing  unitree@192.168.123.15:go1_gym/logs/jenkins_experiment/ navigation/robot_demos/jenkins_experiment/{self.demo_folder}/runs/')

    

    def extract_demos(self):
        print('Extracting data')
      
        log_files = sorted([str(p) for p in Path(self.log_path).glob("*.pkl")],key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))
        for log in log_files:
            with gzip.open(log, 'rb') as f:
                p = pickle.Unpickler(f)
                demo = p.load()

            for k in demo:
                if k=='Time':
                    self.demos[k]+=demo[k]
                else:
                    print(len(demo[k]))
                    self.demos[k]+=demo[k]

        self.rgb=self.demos['Image1st']
        #self.depth=self.demos['DepthImg']
        self.commands=self.demos['Commands']

        self.rounded_comms = [[round(com,3) for com in c] for c in self.commands]

        print('Time of demo:', self.demos['Time'])

    def update_demos(self):
        print('Updating demos...demo',self.run_num)
        log_files = sorted([str(p) for p in Path(self.log_path).glob("*.pkl")],key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))

        
        for log in log_files:
            with open(log, 'rb') as f:
                demo = pickle.load(f)

            with gzip.open(log, "wb") as f:
                pickled = pickle.dumps(demo)
                optimized_pickle = pickletools.optimize(pickled)
                f.write(optimized_pickle)
            


if __name__ == "__main__":

    runs = [11]
    for r in runs:
        viewer = DemoViewer(run_num=r, extract =False, save_video=False, demo_folder='simple', use_nn=False, use_rgb=True)
        viewer.view_video()
        #viewer.update_demos()
    

    

    #viewer.view_single()
    #viewer.update_demos()

    #viewer.download_all_data('simple')
