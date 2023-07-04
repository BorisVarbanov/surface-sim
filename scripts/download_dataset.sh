#!/bin/sh

username="bmvarbanov"
dataset="20230314-rot-xzzx-surface_circ-level_threshold"
from_scrath=false



if $from_scrath;
then
    host="login.delftblue.tudelft.nl"
    source_dir="/scratch/${username}/data/${dataset}"
else
    host="linux-bastion.tudelft.nl"
    source_dir="/tudelft.net/staff-umbrella/qrennd/data/${dataset}"
fi

destination_dir="$HOME/data/"

rsync -avzh --exclude "*.ipynb" --progress --stats --no-perms "${username}@${host}:${source_dir}" "${destination_dir}"