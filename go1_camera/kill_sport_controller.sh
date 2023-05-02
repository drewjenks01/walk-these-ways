# @reboot sh /scratch/gmargo/AutoStart/kill_sport_controller.sh


#!/bin/bash
sleep 25
sudo kill $(ps aux | grep sport | awk '{print $2}')