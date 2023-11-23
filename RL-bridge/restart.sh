#!/bin/bash

docker restart fanzl_AAMAS_elo

for i in {2..170}
do
    docker restart fanzl_AAMAS_actor_$i
done
