#!/bin/bash

docker restart fanzl_AAMAS_elo

for i in {2..150}
do
    docker restart fanzl_AAMAS_actor_$i
done
