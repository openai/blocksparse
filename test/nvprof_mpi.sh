#!/bin/bash

if [ "$PMIX_RANK" == "0" ]; then
    nvprof -f -o nccl_test.nvvp python nccl_test.py
else
    python nccl_test.py
fi

# nvprof -f -o "dota_%p.nvvp" --profile-child-processes