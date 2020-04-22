#!/bin/bash

mkdir -p results
cd results

for ensaio in 1 2 3 4 5
do
#	for caso in case14.m case30.m case57.m case118.m case300.m case1354pegase.m case2869pegase.m case9241pegase.m;
	for caso in case1354pegase.m
	do
		for exe in P O S;
		do
			for num in 1 10 50 100 200
			do			
				echo "Starting MAPoL"
				echo "	Case: $caso"
				if [ "$exe" = "P" ]; then
					echo "	Running in: GPU"
				else
					echo "	Running in: CPU"
				fi
				echo "	# of Particles: $num"
				echo "	Execution: $ensaio"
				../Debug/MyMAPoL ../datasets/$caso $exe $num
			done
		done
	done
done
