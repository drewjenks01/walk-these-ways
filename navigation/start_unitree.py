
import os

# starts Unitree SDK -- if LCM not initilized error, reboot robot and try again
print('Starting Unitree SDK...')
os.system('cd ~/go1_gym/go1_gym_deploy/autostart && chmod +x ./start_unitree_sdk.sh && ./start_unitree_sdk.sh')

print('Unitree SDK started.')
print('---------------------')

# starts the foxy controller policy
print('Starting Foxy Controller...')
os.system('cd .. && cd docker')
os.system('sudo make run && cd /home/isaac/go1_gym/ && python3 setup.py install && cd go1_gym_deploy/scripts && python3 deploy_policy.py')
print('Foxy controller started.')
print('---------------------')


