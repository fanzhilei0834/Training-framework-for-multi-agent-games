#!/bin/bash

docker exec -d fanzl_AAMAS_elo /bin/bash -c "python localgame.py"

for i in {2..170}
do
    docker exec -d fanzl_AAMAS_actor_$i /bin/bash -c "python actor.py"
done
