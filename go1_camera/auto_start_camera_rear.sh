#!/bin/sh
sleep 40
cd /home/unitree/go1_gym_dev/go1_camera/UnitreecameraSDK
./bins/putImagetrans_reardown_fisheye &

sleep 10
kill -9 $(ps aux |grep putImage | awk '{print $2}')
cd /home/unitree/go1_gym_dev/go1_camera && export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/site-packages/ && python3 rear_log_lmdb.py &
