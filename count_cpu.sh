#!/bin/bash
function cpuu {
	top -b -n 1 -u divyam123 $1 | awk 'NR>7 { sum += $9; } END { print sum; }' 
}

cpuu