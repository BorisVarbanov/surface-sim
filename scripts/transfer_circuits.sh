#!/bin/sh

dataset="20230510-d3_rot-xzzx-google_circ-level_p0.001_y-bias-scan"

source_dir="$HOME/data/${dataset}"
destination_dir="$HOME/OneDrive/qrennd/data/"

rsync -avzh --exclude "*.ipynb" --exclude "*.py" --exclude "*.nc" --progress --stats --no-perms "${source_dir}" "${destination_dir}"