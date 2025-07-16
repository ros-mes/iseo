#!/bin/bash

# Define all combinations of parameters
separators=("SEP" "SVM")
samplers=("CUT" "SIM")
types=("uncorrelated" "weakly-correlated" "strongly-correlated")
ranges=("1 10" "11 20" "21 30")

count=0

for sep in "${separators[@]}"; do
  for sampler in "${samplers[@]}"; do
    for type in "${types[@]}"; do
      for range in "${ranges[@]}"; do
        IFS=' ' read -r first last <<< "$range"
        session_name="kp_${sep}_${sampler}_${type}_${first}_${last}"
        cmd="python -m test.test_kp --separator $sep --sampler $sampler --type $type --first $first --last $last"
        tmux new-session -d -s "$session_name"
        tmux send-keys -t "$session_name" "$cmd" Enter
        sleep 1
        echo "Launched session $session_name with command: $cmd"
        ((count++))
      done
    done
  done
done

echo "$count sessions launched."