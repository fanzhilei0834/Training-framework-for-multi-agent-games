#!/bin/bash

for i in {2..170}
do
    docker run -itd --name fanzl_AAMAS_actor_$i --cpuset-cpus $(($i+9))-$(($i+9)) -v /home/fanzl/AAMAS2023/RL_PPO/:/home/workplace/ -w /home/workplace/ --shm-size 4G aamas_fanzl:v2 /bin/bash
done

docker run -itd --name fanzl_AAMAS_elo --cpuset-cpus 180-189 -v /home/fanzl/AAMAS2023/RL_PPO/:/home/workplace/ -w /home/workplace/ --shm-size 32G aamas_fanzl:v2 /bin/bash
docker run -itd --name AAMAS_fanzl --gpus all --cpuset-cpus 0-9 -v /home/fanzl/AAMAS2023/RL_PPO/:/home/workplace/ -w /home/workplace/ --shm-size 256G aamas_fanzl:v2 /bin/bash