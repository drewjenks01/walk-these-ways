sleep 30
cd /home/unitree/go1_gym_dev/go1_camera
sudo ./prepare_cameras.sh &

# uncomment the following line to autostart low level control on 192.168.123.15
sleep 5
cd /home/unitree/go1_gym_dev/go1_gym_deploy/unitree_legged_sdk_bin
yes "" | sudo ./lcm_position &
cd /home/unitree/go1_gym_dev/go1_gym_deploy/docker && sudo make start_foxy
sudo docker exec foxy_controller bash -c 'cd /home/isaac/go1_gym && python3 setup.py install && cd go1_gym_deploy/scripts && ls && python3 deploy_policy.py'
