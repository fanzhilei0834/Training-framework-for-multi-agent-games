#!/bin/bash

for i in {1..170}
do
    docker rm -f fanzl_AAMAS_actor_$i
done

docker rm -f fanzl_AAMAS_elo
docker rm -f AAMAS_fanzl