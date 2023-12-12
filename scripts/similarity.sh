> output.txt

for seed in $(cat random_seeds.txt); do
    for distance in $(seq 1 2 20); do
        python utils/generate_map.py --seed $seed --target_distance $distance > /dev/null 2>&1

        if [ $? -ne 0 ]; then
            echo "Seed $seed - distance $distance failed: generate_map.py exited with status $?" >> output.txt
            continue
        fi

        echo "Seed $seed - distance $distance: Single task" >> output.txt
        python main.py --seed $seed --map_i=0 > /dev/null 2>&1 # single task 

        echo "Seed $seed - distance $distance: Multi task" >> output.txt
        python main.py --seed $seed --map_i=-1 > /dev/null 2>&1 # multi task
    done
done

