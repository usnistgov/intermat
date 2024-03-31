include init.mod
include potential.mod
#fix 3 all nph aniso 0.0 0.0 1.0
#fix langevin all langevin 0.0 0.0 1.0 12233 

#fix 3 all box/relax tri 0.0 vmax 0.001
#fix 3 all box/relax  aniso 0.0
#dump            2a all cfg 100 *.cfg mass type xs ys zs  vx vy vz fx fy fz
#dump_modify     2a element  Mo S
#fix             2 all qeq/reax 1 0.0 10.0 1e-6 /scratch/lfs/kamal/JARVIS/All2/ReaxFF/param.qeq

#fix            3 all box/relax aniso 0.0 vmax 0.001
#fix 1a all qeq/comb 1 0.0001 file fq.out
#minimize ${etol} ${ftol} ${maxiter} ${maxeval}
write_data data0
run 0

