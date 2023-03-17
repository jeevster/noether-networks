#!/bin/bash

CUDA=(6 7 8 9)
NPAST=(2 2 2 2)
NFUTURE=(2 5 10 20)
BS=(10 10 6 3)
conv_emb
pde_emb


for i in ${!NPAST[@]}; do
    # Tailoring pass then no tailor
    (./arg_run.sh \
    -e "pde_emb" \
    -b ${BS[$i]} \
    -t \
    -c ${CUDA[$i]} \
    ${NPAST[$i]} \
    ${NFUTURE[$i]} > ./bash_logs/${NPAST[$i]}_${NFUTURE[$i]}_tailor_pdeemb.txt 2>&1 && \
    ./arg_run.sh \
    -e "conv_emb" \
    -b ${BS[$i]} \
    -t \
    -c ${CUDA[$i]} \
    ${NPAST[$i]} \
    ${NFUTURE[$i]} > ./bash_logs/${NPAST[$i]}_${NFUTURE[$i]}_tailor_convemb.txt 2>&1 && \
    ./arg_run.sh -b ${BS[$i]} \
    -c ${CUDA[$i]} \
    ${NPAST[$i]} \
    ${NFUTURE[$i]} > ./bash_logs/${NPAST[$i]}_${NFUTURE[$i]}.txt 2>&1) &
done
