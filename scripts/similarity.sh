for seed in $(cat random_seeds.txt); do
    for distance in $(seq 1 2 20); do
        python generate_map.py --seed $seed --target_distance $distance

        if [ $? -ne 0 ]; then
            echo "Seed $seed - distance $distance failed: generate_map.py exited with status $?"
            continue
        fi

        echo "Seed $seed - distance $distance: Single task"
        python main.py --seed $seed --map_i=0 # single task

        echo "Seed $seed - distance $distance: Multi task"
        python main.py --seed $seed --map_i=-1  # multi task
    done
done

