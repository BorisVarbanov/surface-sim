#!/bin/sh

username="bmvarbanov"
dataset="20230413-d3_rot-css-surface_circ-level_p0.001_y-bias-scan"
to_scrath=true



if $to_scrath;
then
    host="login.delftblue.tudelft.nl"
    destination_dir="/scratch/${username}/data/"
else
    host="linux-bastion.tudelft.nl"
    destination_dir="/tudelft.net/staff-umbrella/qrennd/data/"
fi

source_dir="$HOME/data/${dataset}"

rsync -avzh --exclude "*.ipynb" --progress --stats --no-perms "${source_dir}" "${username}@${host}:${destination_dir}"