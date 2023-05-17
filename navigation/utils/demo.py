#%%
import time
import os
import pickle as pkl
from pathlib import Path
import gzip
import pickletools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from navigation.utils.image_processing import process_deployed, process_depth
from navigation.commandnet.commandNN import CommandNet
import numpy as np
import shutil

class Demo:

    def __init__(self,log_root , log_view_folder=None,extract=False, use_nn = False, save_video = False, use_rgb_viewing=False, run_num = None):

        # log attributes
        self.log_root = log_root
        self.log_save_path = self.log_root+'/curr_run/'
        self.log_filename=''
        self.log_count=1
        self.log_iter_count=1
        self.log_view_path= f'{self.log_root}/{log_view_folder}/run{run_num}'

        # viewing demo
        self.use_nn = use_nn
        self.save_video = save_video
        self.use_rgb_viewing = use_rgb_viewing
        self.demos = {}
        self.run_num = run_num
        self.log_path = f'{self.log_save_path}run{run_num}'
        self.extract = extract

        self.fps = 6

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

        self.prepare_flag=False

        # logging bool
        self.logging = False


        # x, y, yaw
        self.frame_commands = {}


    def init_log(self, start_iter,same_name=False):
        
        # reset log count
        self.log_count = 1

        self.start_iter=start_iter

        # reset log iters
        self.log_iter_count = 1

        self._reset_log()

        if not same_name:
            # make base log filenname
            num_runs = len(os.listdir(self.log_save_path))
            curr_run = num_runs+1
            os.makedirs(self.log_save_path+f'/run{curr_run}')
            self.log_filename = self.log_save_path+f'/run{curr_run}/'

            print(f'{num_runs} previous logs')

        # turn logging on
        self.logging = True

        # start log time
        self.log_start_time = time.time()

        print('Starting log...')

    def save_partial_log(self, time_val=None):
        # make partial log filename
        log_name = self.log_filename+f'log{self.log_count}.pkl'

        # update log count
        self.log_count+=1

        if time_val:
            self.log['Time']=time_val

        # fps = len(self.log['Image1st'])/self.log['Time']
        # print('FPS:',fps)

        # update log start time
        self.log_start_time = time.time()

        # save log
        print('Saving partial log...')
        with open(log_name, 'wb') as file:
            pkl.dump(self.log, file, pkl.HIGHEST_PROTOCOL)  

        print('Partial log saved to', log_name)

        # reset log
        self._reset_log()

    def end_log(self, end_iter, no_save=False):
        # set logging to false
        self.logging = False

        # save last log
        if not no_save:
            self.save_partial_log(time_val=end_iter-self.start_iter)

        else:

            self._reset_log()

        print('Log ended.')



    def collect_demo_data(self, data):
        # data should be dict

        # avg comms
        avg_comms = []

        #print('demo log:', self.log)

        for k in self.frame_commands:
            if k != 'count':
                avg_comms.append(self.frame_commands[k]/self.frame_commands['count'])
        self.log['Commands'].append(avg_comms)

        for k in data:
            self.log[k].append(data[k])

        self._reset_frame_commands()

        self.log_iter_count+=1

        print('Collected data count:', len(self.log['Commands']))

        time_val = time.time()-self.log_start_time
        if time_val>=5:
            self.save_partial_log()
    
    def extract_from_robot(self):
        print('Extracting from robot')
        os.system(f'rsync -a -P --info=progress2 --ignore-existing  unitree@192.168.123.15:go1_gym/logs/jenkins_experiment/ navigation/robot_demos/jenkins_experiment/')

    
    def extract_demos(self, path=None, plot=False, key=None):
        print('Extracting data')
        self._reset_log()

        if not path:
            path = self.log_view_path

        print(path)

      
        log_files = sorted([str(p) for p in Path(path).glob("*.pkl")],key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))

        if plot:
            log_files = log_files[::-1]

        demo_success = None
       # print(log_files)
        for log in log_files:
            with gzip.open(log, 'rb') as f:
                p = pkl.Unpickler(f)
                demo = p.load()
                #print(demo.keys())

            if plot:
                if not demo_success:
                    demo_success=demo['Success']

                if demo_success==1.0:
                    for k in demo:
                        #print(demo['Success'])
                        self.log[k]+=demo[k]
            else:
                for k in demo:
                    self.log[k]+=demo[k]

        rgb=self.log['Image1st']
        if self.log['DepthImg']:
            depth=self.log['DepthImg']
        else:
            depth=[]
        commands=self.log['Commands']

        print(f'Comm: {len(self.log["Commands"])}, RGB: {len(self.log["Image1st"])}, Depth: {len(self.log["DepthImg"])} ')

        rounded_comms = [[round(com,3) for com in c] for c in commands]

        t = np.array(self.log['Torque'])
        jv = np.array(self.log['Joint_Vel'])
        energy = np.sum(t*jv)
        self.log['Energy_time']=[sum(t[i]*jv[i]) for i in range(len(t))]
        print('Time of demo:', self.log['Time'], 'Distance traveled: ',sum(self.log['Distance']), 'Success: ', self.log['Success'], 'Energy: ', energy, 'Avg drift:', np.average(self.log['Drift']))

        if plot:
            return [self.log['Success'], self.log['Time'], energy], self.log['Commands']
        elif key:
            return self.log[key]
        else:
            return rgb, rounded_comms, depth

    def collect_frame_commands(self, comms):
        self.frame_commands['count']+=1
        for k in comms:
            self.frame_commands[k]+=comms[k]

    def view_single(self,indx=None, save_rgb=False):

        rgb, rounded_comms = self.extract_demos()
        
        fig, ax = plt.subplots(1,1)
        if not indx:
            import random
            indx = random.randint(0,len(rgb)-1)

          
        # ax[0].imshow(rgb[indx])
        # ax[0].set_title('RGB')
        # # ax[1].imshow(self.depth[indx])
        # # ax[1].set_title('Depth')
        # #fig.suptitle(f'Command: {self.rounded_comms[indx]}')
        # plt.show()

        if save_rgb:
            from PIL import Image
            im = Image.fromarray(rgb[indx])
            im.save("example_robot_rgb.jpeg")


    def view_video(self, just_vid=False):
        import cv2
        from torchvision import transforms
        from tqdm import tqdm
        to_pil=transforms.Compose([
        transforms.ToPILImage()]
    )
        rgb, rounded_comms, depth = self.extract_demos()
        print(depth)

        print('Preparing video...')
        if just_vid:
            num_plots=1
        else:
            num_plots=3

        fig,ax=plt.subplots(1,num_plots)
        fig.set_size_inches(20,10)
        fig.suptitle(f'Run{self.run_num}')


        if not just_vid:
            ax[0].set_title('RGB')
            ax[1].set_title('Depth')

            if self.use_nn:
                ax[2].text(0.01, 0.5, 'Pred Commands: \n True Commands: ')
            else:
                ax[2].text(0.01, 0.5, 'Commands: ')
        else:
            ax.set_title('RGB')

        movie=[]
        for f in tqdm(range(len(rgb))):

            img_rgb = rgb[f]
            img= process_deployed(img_rgb)[0]
            if depth:
                img_depth = depth[f]

            
            if self.use_nn:
                commands, policy = self.model(img)
                commands, policy = self.model._data_rescale(commands, policy)

                commands.append(policy)
                commands = [round(c,4) for c in commands]
                m=[]

    

                # depth, commands
                if not just_vid:
                    m += [ax[0].imshow(to_pil(img),animated=True)]

                    m+=[ax[1].imshow(process_depth(img_depth),animated=True)]

                    m+=[ax[2].text(0.01, 0.5, f'Pred Commands: {commands}\n True Commands: {rounded_comms[f]}')]
                
                else:
                    # rgb
                    m += [ax.imshow(to_pil(img),animated=True)]


                movie.append(m)


            else:

                if self.use_rgb_viewing:
                    show_img=img_rgb
                else:
                    show_img=to_pil(img)

                m=[]

                # depth, commands
                if not just_vid:
                    m += [ax[0].imshow(to_pil(show_img),animated=True)]

                    m+=[ax[1].imshow(process_depth(img_depth),animated=True)]

                    m+=[ax[2].text(0.01, 0.5, f'Commands: {rounded_comms[f]}')]

                else:
                    m += [ax.imshow(to_pil(show_img),animated=True)]


                movie.append(m)
        
        ani = animation.ArtistAnimation(fig,movie,interval=100,blit=True)

        if self.save_video:
            ani.save(f'video_run{self.run_num}.mp4',fps=8,bitrate=-1,codec="libx264")
            print('Video saved.')
        plt.show()

    
    def update_demos(self):
        print('Updating demos...demo',self.run_num)
        log_files = sorted([str(p) for p in Path(self.log_path).glob("*.pkl")],key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))

        try:
            for log in log_files:
                with open(log, 'rb') as f:
                    demo = pkl.load(f)

                with gzip.open(log, "wb") as f:
                    pickled = pkl.dumps(demo)
                    optimized_pickle = pickletools.optimize(pickled)
                    f.write(optimized_pickle)

            print('Updated!')
        
        except:
            print('Already updated')


    def post_process_success(self):
        print('Updating demos...demo',self.run_num)
        log_files = sorted([str(p) for p in Path(self.log_path).glob("*.pkl")],key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))

         # watch video of log
        self.view_video()

        print('\n')
        inp = input('Test success?')

        # check if test is sucess
        if inp=='y':
            success=1
        else:
            success=0

        
        for log in log_files:

            # load log
            with gzip.open(log, 'rb') as f:
                p = pkl.Unpickler(f)
                demo = p.load()

            #update log
            demo['Success'] = success
            
            with gzip.open(log, "wb") as f:
                pickled = pkl.dumps(demo)
                optimized_pickle = pickletools.optimize(pickled)
                f.write(optimized_pickle)

    def create_performance_plot(self):
        from copy import deepcopy


        metrics=['Success', 'Time', 'Energy']

        log = {m:0 for m in metrics}

        # load WTW runs
        # log_path = self.log_root+'WTW/'
        # num_runs = len(os.listdir(log_path))

        # wtw = deepcopy(log)
        # for i in range(num_runs):
        #     outs,_ = self.extract_demos(f'{log_path}run{i+1}', plot=True)

        #     for i in range(len(outs)):
        #         wtw[metrics[i]]+=outs[i]

        # for k in wtw:
        #     wtw[k]/=num_runs

         # load stair runs
        log_path = self.log_root+'stair/'
        num_runs = len(os.listdir(log_path))

        stair = deepcopy(log)
        for i in range(num_runs):
            outs,_ = self.extract_demos(f'{log_path}run{i+1}', plot=True)
            
            for i in range(len(outs)):
                stair[metrics[i]]+=outs[i]

        succ_amt = stair['Success']
        stair['Success']/=num_runs
        for k in stair:
            if k !='Success':
                stair[k]/=succ_amt



         # load combo runs
        log_path = self.log_root+'comb_trial/'
        num_runs = len(os.listdir(log_path))

        combo = deepcopy(log)
        combo_policies = {0:0,1:0}
        for i in range(num_runs):
            outs,comms = self.extract_demos(f'{log_path}run{i+1}', plot=True)
            
            for i in range(len(outs)):
                combo[metrics[i]]+=outs[i]

            #print(comms)
            for i in range(len(comms)):
                #print(comms[i])
                combo_policies[round(comms[i][3])]+=1
        #print(combo_policies)
        # combo_policies[0]/=num_runs
        # combo_policies[1]/=num_runs
        
        succ_amt = combo['Success']
        combo['Success']/=num_runs
        for k in combo:
            if k !='Success':
                combo[k]/=succ_amt

         # load auto runs
        log_path = self.log_root+'auto/'
        num_runs = len(os.listdir(log_path))

        auto = deepcopy(log)
        auto_policies = {0:0,1:0}
        for i in range(num_runs):
            outs,comms = self.extract_demos(f'{log_path}run{i+1}', plot=True)
            
            for i in range(len(outs)):
                auto[metrics[i]]+=outs[i]

            for i in range(len(comms)):
                auto_policies[round(comms[i][3])]+=1

            #print(auto_policies)
        succ_amt = auto['Success']
        auto['Success']/=num_runs
        for k in combo:
            if k !='Success':
                auto[k]/=succ_amt


         # load auto-climb runs
        log_path = self.log_root+'auto_climb/'
        num_runs = len(os.listdir(log_path))

        auto_climb = deepcopy(log)
        for i in range(num_runs):
            outs,comms = self.extract_demos(f'{log_path}run{i+1}', plot=True)
            
            for i in range(len(outs)):
                auto_climb[metrics[i]]+=outs[i]

        succ_amt = auto_climb['Success']
        auto_climb['Success']/=num_runs
        for k in combo:
            if k !='Success':
                auto_climb[k]/=succ_amt


        # auto_policies[0]/=num_runs
        # auto_policies[1]/=num_runs

        #print(wtw)
       # Create the figure and subplots
        x = ['climb', 'combo', 'ours (climb only)', 'ours']
        y = []
        for k in metrics:
            if k == 'Success':
                sub= [100*stair[k], 100*combo[k], 100*auto_climb[k],100*auto[k]]
            else:
                sub= [stair[k], combo[k], auto_climb[k],auto[k]]
            y.append(sub)

        # Create the figure and subplots
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (8,5),sharey=False)
        fig.supxlabel('Policy Method')
        colors = ['r','g','b','orange']

        for i in range(3):
            for k in range(len(x)):
                label = x[k]

                axs[i].bar(0.2*k,y[i][k], width = 0.2, color = colors[k],label=label)

        # Plot the data on each subplot
        # axs[0, 0].bar(x, y[0], width=0.4, color=colors)
        # axs[0, 1].bar(x, y[1], width=0.4, color=colors)
        # axs[1, 0].bar(x, y[2], width=0.4, color=colors)
        # axs[1, 1].bar(x, y[3], width=0.4, color=colors)

        # Set the y-axis scales for each subplot
        axs[0].set_ylim([0, 100])
        axs[1].set_ylim([0, 3200])
        axs[2].set_ylim([0, 35000])
       # axs[1, 1].set_ylim([0, 40])

        # Set titles, labels, and ticks for each subplot
        axs[0].set_title('Success')
        axs[1].set_title('Time')
        axs[2].set_title('Energy')
        #axs[1, 1].set_title('Distance')
        # axs[0, 0].set_xlabel('X-axis')
        # axs[0, 1].set_xlabel('X-axis')
        # axs[1, 0].set_xlabel('X-axis')
        # axs[1, 1].set_xlabel('X-axis')
        axs[0].set_ylabel('%')
        axs[1].set_ylabel('Iters')
        axs[2].set_ylabel('Joules')
        #axs[1, 1].set_ylabel('Meters')

        axs[0].set_xticks([])
        axs[1].set_xticks([])
        axs[2].set_xticks([])
        # axs[0, 0].set_xticklabels(x, rotation=0)
        # axs[0, 1].set_xticklabels(x, rotation=0)
        # axs[1, 0].set_xticklabels(x, rotation=0)
        # axs[1, 1].set_xticklabels(x, rotation=0)
        # Add some padding between subplots
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles= handles, labels=labels,bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.tight_layout()
        

        # Save the figure as a PNG
        plt.savefig('navigation/utils/plots/human_demo_compare.pdf', dpi=300, format='pdf',bbox_inches='tight')
        plt.savefig('navigation/utils/plots/human_demo_compare.jpg', dpi=300, format='jpg',bbox_inches='tight')
       # plt.show()

        # plot of policies for combo and auto
        x = ['walk', 'climb']
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6,6), sharey=False)
        comb_tot = combo_policies[0]+combo_policies[1]
        combo_perc = [100*combo_policies[0]/comb_tot, 100*combo_policies[1]/comb_tot]

        auto_tot = auto_policies[0]+auto_policies[1]
        auto_perc = [100*auto_policies[0]/auto_tot, 100*auto_policies[1]/auto_tot]

        X_axis = np.arange(len(x))
  
        plt.bar(X_axis - 0.2, combo_perc, 0.4, label = 'combo')
        plt.bar(X_axis + 0.2, auto_perc, 0.4, label = 'auto')
        
        plt.xticks(X_axis, x)
        plt.xlabel("Navigation Method")
        plt.ylabel("%")
        plt.title("Percentage of each policy used during traversal")
        plt.legend()
        plt.savefig('navigation/utils/plots/policy_perc.jpg', dpi=300, format='jpg')
    
    def log_key_plot(self,key):

        log = self.extract_demos(key=key)

        plt.figure()
        plt.plot(range(len(log)), log)
        plt.title(f'{self.log_view_path}, {key}')
        plt.savefig('navigation/utils/plots/log_key.jpg', dpi=300, format='jpg')


    
    def undo_log(self):
        shutil.rmtree(self.log_filename)
    
    
    def _reset_log(self):
        self.log = {'Commands':[], 'Image1st':[], 'DepthImg':[], 'Time':0, 'Torque':[], 'Joint_Vel':[], 'Success':0, 'Distance':[],'Drift':[]}
        self._reset_frame_commands()

    def _reset_frame_commands(self):
        self.frame_commands = {'x':0, 'y':0,'yaw':0,'policy':0,'count':0}



if __name__ == "__main__":

    log_root = f'navigation/robot_demos/jenkins_experiment/'
    #log_root = f'navigation/robot_demos/icra_trials/'
    log_view_folder = 'curtain'

    
    extract = False
    save_video = False

    use_nn = False
    use_rgb = True

    runs = [1,2,3,4,5,6,7,8,9,10]
    #runs=[1]
    
    for run in runs:
        demo = Demo(log_root=log_root, log_view_folder = log_view_folder,run_num=run, extract = extract, save_video=save_video, use_nn=use_nn, use_rgb_viewing=use_rgb)
        demo.update_demos()
        demo.view_video(just_vid=True)

        #demo.view_single(indx=0, save_rgb=True)
        #demo.create_performance_plot()
        #demo.log_key_plot('Energy_time')

# %%
