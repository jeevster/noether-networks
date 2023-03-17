#!/bin/bash

CUDA=(6 7 8 9)
NPAST=(2 2 2 2)
NFUTURE=(2 5 10)
BS=(10 10 6)


for i in ${!NPAST[@]}; do
    # Tailoring pass then no tailor
    (./argrun.sh \
    -b ${BS[$i]} \
    -t \
    -c ${CUDA[$i]} \
    ${NPAST[$i]} \
    ${NFUTURE[$i]} > ./bash_logs/${NPAST[$i]}_${NFUTURE[$i]}_tailor.txt 2>&1 && \
    ./argrun.sh -b ${BS[$i]} \
    -c ${CUDA[$i]} \
    ${NPAST[$i]} \
    ${NFUTURE[$i]} > ./bash_logs/${NPAST[$i]}_${NFUTURE[$i]}.txt 2>&1) &
done
