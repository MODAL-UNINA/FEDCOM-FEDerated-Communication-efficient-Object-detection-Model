#!/bin/bash

FRAMEWORK=${1:-"FEDCOM"}
SCENARIO=${2:-"3"}
DOMAIN_ID=${3:-"2"}
SERVER_GPU_ID=${4:-"0"}
CLIENTS_GPU_ID=${5:-"1,2,3"}
ROUNDS_PER_DOMAIN=${6:-"51"}
EPOCHS=${7:-"5"}
SERVER_ADDRESS=${8:-"0.0.0.0:8080"}
SIMILARITY_THRESHOLD=${9:-"0.75"}
MAX_IMAGES=${10:-"20"}
MODEL_NAME=${11:-"yolo12s_upd.yaml"}

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: CONDA_PREFIX is not set. Please activate your conda environment."
    exit 1
fi

echo "Running experiment with the following parameters:" \
    "FRAMEWORK: $FRAMEWORK, " \
    "SCENARIO: $SCENARIO, " \
    "DOMAIN_ID: $DOMAIN_ID, " \
    "SERVER_GPU_ID: $SERVER_GPU_ID, " \
    "CLIENTS_GPU_ID: $CLIENTS_GPU_ID, " \
    "ROUNDS_PER_DOMAIN: $ROUNDS_PER_DOMAIN, " \
    "EPOCHS: $EPOCHS, " \
    "SERVER_ADDRESS: $SERVER_ADDRESS, " \
    "SIMILARITY_THRESHOLD: $SIMILARITY_THRESHOLD, " \
    "MAX_IMAGES: $MAX_IMAGES, " \
    "MODEL_NAME: $MODEL_NAME"

CLIENTS_GPU_IDS=(${CLIENTS_GPU_ID//,/ })

MAX_NUM_CLIENTS=${#CLIENTS_GPU_IDS[@]}

mkdir -p ../Results/logs

pids=()

# Server
$CONDA_PREFIX/bin/python -uB 1_main_server.py \
    --framework=$FRAMEWORK \
    --model-name=$MODEL_NAME \
    --scenario=$SCENARIO \
    --domain-id=$DOMAIN_ID \
    --server-address=$SERVER_ADDRESS \
    --max-num-clients=$MAX_NUM_CLIENTS \
    --rounds-per-domain=$ROUNDS_PER_DOMAIN \
    --gpu-id=$SERVER_GPU_ID \
    > ../Results/logs/run_logs_${FRAMEWORK}_scenario_${SCENARIO}_domain_${DOMAIN_ID}_server.txt 2>&1 & pids+=($!)

sleep 10  # Give the server some time to start 

# Clients
for i in $(seq 0 $((MAX_NUM_CLIENTS - 1))); do
    CLIENT_GPU_ID=${CLIENTS_GPU_IDS[$i]}
    $CONDA_PREFIX/bin/python -uB 1_main_client.py \
        --framework=$FRAMEWORK \
        --model-name=$MODEL_NAME \
        --scenario=$SCENARIO \
        --domain-id=$DOMAIN_ID \
        --server-address=$SERVER_ADDRESS \
        --max-num-clients=$MAX_NUM_CLIENTS \
        --epochs=$EPOCHS \
        --similarity-threshold=$SIMILARITY_THRESHOLD \
        --max-images=$MAX_IMAGES \
        --client-id=$i \
        --gpu-id=$CLIENT_GPU_ID \
        > ../Results/logs/run_logs_${FRAMEWORK}_scenario_${SCENARIO}_domain_${DOMAIN_ID}_client_$i.txt 2>&1 & pids+=($!)
done

echo "PIDS: ${pids[@]}"

# rets=()
# for pid in ${pids[*]}; do
#     wait $pid
#     rets+=($?)
# done
while (( ${#pids[@]} )); do
    for pid_idx in "${!pids[@]}"; do
        pid=${pids[$pid_idx]}
        if ! kill -0 "$pid" 2>/dev/null; then # kill -0 checks for process existance
            # we know this pid has exited; retrieve its exit status
            wait "$pid" || { kill "${pids[@]}"; echo "PID $pid failed. Exiting"; exit 1; }
            rets+="($pid_idx -> $?)"
            unset "pids[$pid_idx]"
        fi
    done
    sleep 1 # in bash, consider a shorter non-integer interval, ie. 0.2
done

echo "Return codes: ${rets[*]}"
