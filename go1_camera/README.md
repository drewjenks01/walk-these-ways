# Go1 robot camera 


## Set up Autostart

The script at `cd walk-these-ways-dev/go1_gym_deploy/scripts/ && ./send_to_unitree.sh` will send the appropriate files to all computers when connected via ethernet.

Then, for the **head (`192.168.123.13`)**:

add to `sudo crontab -e`:
```bash
@reboot sh /home/unitree/go1_gym_deploy/go1_camera/auto_start_prepare.sh
```
add to `crontab -e`:
```bash
@reboot sh /home/unitree/go1_gym_deploy/go1_camera/auto_start_camera_head.sh -i $INDEX 2>&1 | /home/unitree/debug_cam.log
```

Then, for the **body (`192.168.123.14`)**:

add to `sudo crontab -e`:
```bash
@reboot sh /home/unitree/go1_gym_deploy/go1_camera/auto_start_prepare.sh
```
add to `crontab -e`:
```bash
@reboot sh /home/unitree/go1_gym_deploy/go1_camera/auto_start_camera_body.sh -i $INDEX 2>&1 | /home/unitree/debug_cam.log
```

Then, for the **rear (`192.168.123.15`, board where the controller will run)**:

add to `sudo crontab -e`:
```bash
@reboot sh /home/unitree/go1_gym_deploy/go1_camera/auto_start_prepare_rear.sh
```
add to `crontab -e`:
```bash
@reboot sh /home/unitree/go1_gym_deploy/go1_camera/auto_start_camera_rear.sh -i $INDEX 2>&1 | /home/unitree/debug_cam.log
```