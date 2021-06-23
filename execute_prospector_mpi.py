from run_prospector_latest_mpi import run_prospector
#import os

#os.system('mpirun -np 8 python run_prospector_latest_mpi.py')

#run_prospector(1)
for i in range(2,50):
    a = run_prospector(i)
    print(a)
    if a == None:
        continue
