#!/bin/bash
kill $(ps aux |grep autostart | awk '{print $2}')
kill -9 $(ps aux |grep ros_ | awk '{print $2}')
kill -9 $(ps aux |grep get_ | awk '{print $2}')
kill -9 $(ps aux |grep put | awk '{print $2}')
kill -9 $(ps aux |grep nx_ | awk '{print $2}')
kill -9 $(ps aux |grep point | awk '{print $2}')
kill -9 $(ps aux |grep mqttControl | awk '{print $2}')

# setup ipv4 multicast settings for LCM
sudo ifconfig eth0 multicast

runtime="5 minute"
endtime=$(date -ud "$runtime" +%s)

while [[ $(date -u +%s) -le $endtime ]]
do
    sudo route del -net 224.0.0.0 netmask 240.0.0.0 dev lo
    sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev eth0
    sleep 1
done