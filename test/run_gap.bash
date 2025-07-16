#!/bin/bash

# Define all combinations of parameters
separators=("SEP" "SVM")
samplers=("CUT" "SIM")
agents=("5" "8" "10")
ranges=("1 5")

count=0

for sep in "${separators[@]}"; do
  for sampler in "${samplers[@]}"; do
    for ag in "${agents[@]}"; do
      for range in "${ranges[@]}"; do
        IFS=' ' read -r first last <<< "$range"
        session_name="gap_${sep}_${sampler}_${ag}_${first}_${last}"
        cmd="python -m test.test_gap --separator $sep --sampler $sampler --agents $ag --first $first --last $last"
        tmux new-session -d -s "$session_name"
        tmux send-keys -t "$session_name" "$cmd" Enter
        echo "Launched session $session_name with command: $cmd"
        ((count++))
      done
    done
  done
done

echo "$count sessions launched."