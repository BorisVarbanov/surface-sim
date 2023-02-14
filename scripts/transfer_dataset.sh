#!/bin/sh

username="bmvarbanov"
dataset="20230214-d3_rot-surf_circ-level_p0.001"
to_scrath=false



if $to_scrath;
then
    host="login.delftblue.tudelft.nl"
    destination_dir="/scratch/${username}/data/"
else
    host="linux-bastion.tudelft.nl"
    destination_dir="/tudelft.net/staff-umbrella/qrennd/data/"
fi

script_dir=$PWD
source_dir="${script_dir}/${dataset}"

rsync -avzh --exclude "*.ipynb" --progress --stats --no-perms "${source_dir}" "${username}@${host}:${destination_dir}"