#!/bin/bash
# download docker image if it doesn't exist yet
wget --directory-prefix=../docker -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XkVpyYyYqQQ4FcgLIDUxg-GR1WI89-XC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XkVpyYyYqQQ4FcgLIDUxg-GR1WI89-XC" -O deployment_image.tar && rm -rf /tmp/cookies.txt

#rsync -av -e ssh --exclude=*.pt --exclude=*.mp4 $PWD/../../go1_gym_deploy $PWD/../../runs $PWD/../../setup.py pi@192.168.12.1:/home/pi/go1_gym
rsync -av -e ssh --exclude=*.pt --exclude=*.mp4 --exclude *.tar --exclude *.urdf --exclude *.pkl --exclude *.stl --exclude *.xacro $PWD/../../go1_gym_deploy $PWD/../../go1_camera $PWD/../../runs $PWD/../setup.py unitree@192.168.123.15:/home/unitree/go1_gym
rsync -av -e ssh --exclude=*.pt --exclude=*.mp4 --exclude *.tar --exclude *.urdf --exclude *.pkl --exclude *.stl --exclude *.xacro $PWD/../../navigation unitree@192.168.123.15:/home/unitree/go1_gym/go1_camera
rsync -av -e ssh --exclude=*.pt --exclude=*.mp4 --exclude *.tar --exclude *.urdf --exclude *.stl --exclude *.xacro $PWD/../../navigation/commandnet/runs unitree@192.168.123.15:/home/unitree/go1_gym/go1_camera/navigation/commandnet

#rsync -av -e ssh --exclude=*.pt --exclude=*.mp4 --exclude *.tar $PWD/../../go1_camera unitree@192.168.123.15:/home/unitree/go1_gym_dev
# rsync -av -e ssh --exclude=*.pt --exclude=*.mp4 --exclude *.tar $PWD/../../go1_camera unitree@192.168.123.14:/home/unitree/go1_gym_dev
#rsync -av -e ssh --exclude=*.pt --exclude=*.mp4 --exclude *.tar $PWD/../../go1_camera unitree@192.168.123.13:/home/unitree/go1_gym_dev
#scp -r $PWD/../../runs pi@192.168.12.1:/home/pi/go1_gym
#scp -r $PWD/../../setup.py pi@192.168.12.1:/home/pi/go1_gym
