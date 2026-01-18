#!/usr/bin/env fish

# Get the directory where this script is located
set script_dir (dirname (status filename))
conda activate GameAdvisor

while true
    # Call the Python script and wait for it to finish
    python3 $script_dir/create_game_dataset.py
    
    # Wait 10 minutes before running again
    echo "Waiting 10 minutes before next execution..."
    sleep 600
end
