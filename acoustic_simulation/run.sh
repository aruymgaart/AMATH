#--- Homogeneous ---
#python3 acoustic_simulation.py SQUARE:100:NEUMANN   1001 0.0000003077 T 200 GAUSS:0.25:0.25  NONE:0:0:0
#python3 acoustic_simulation.py SQUARE:100:DIRICHLET 1001 0.0000003077 T 200 GAUSS:0.25:0.25  NONE:0:0:0
#python3 acoustic_simulation.py SQUARE:100:ABSORBING 1001 0.0000003077 T 200 GAUSS:0.25:0.25  NONE:0:0:0
#python3 acoustic_simulation.py SQUARE:100:PERIODIC  1001 0.0000003077 T 200 GAUSS:0.25:0.25  NONE:0:0:0
python3 acoustic_simulation.py SQUARE:128:PERIODIC  1001 0.000001 T 200 GAUSS:0.25:0.25  NONE:0:0:0 SPECTRAL
#python3 acoustic_simulation.py SQUARE:128:PERIODIC  1001 0.000001 T 200 GAUSS:0.25:0.25  NONE:0:0:0 FINDIFF
#--- Nonhomogeneous: Gaussian source ---
#python3 acoustic_simulation.py SQUARE:100:DIRICHLET 2001 0.0000003077 T 200 NONE:0.0:0.0     GAUSS:0.5:0.5:0.003
#python3 acoustic_simulation.py SQUARE:100:NEUMANN   2401 0.0000003077 T 200 NONE:0.0:0.0     GAUSS:0.5:0.5:0.003
#python3 acoustic_simulation.py SQUARE:100:PERIODIC  2001 0.0000003077 T 200 NONE:0.0:0.0     GAUSS:0.5:0.5:0.003
#python3 acoustic_simulation.py SQUARE:100:ABSORBING 2401 0.0000003077 T 200 NONE:0.0:0.0     GAUSS:0.5:0.5:0.003

#python3 acoustic_simulation.py FILE:BC.helmholtz_1.ini 5001 0.000002 T 100 RECT:60:179:1:199:0.05  NONE:0:0:0
#python3 acoustic_simulation.py FILE:BC.helmholtz_2.ini 5001 0.000002 T 100 RECT:60:179:1:199:0.05  NONE:0:0:0
#python3 acoustic_simulation.py FILE:BC.helmholtz_1.ini 1001 0.0006 T 100 GAUSS:0.7:0.5  NONE:0:0:0
#python3 acoustic_simulation.py FILE:BC.helmholtz_1.ini 3001 0.0006 T 100 NONE:0.0:0.0  GAUSS:0.1:0.5:0.001

#python3 acoustic_simulation.py FILE:BC.helmholtz_1.ini 1001 0.0006 T 100 GAUSS:0.1:0.1  NONE:0:0:0
#python3 acoustic_simulation.py FILE:BC.helmholtz_1.ini 1001 0.0006 T 100 GAUSS:0.5:0.7  NONE:0:0:0
#python3 acoustic_simulation.py FILE:BC.helmholtz_1.ini 1001 0.0006 T 100 NONE:0.0:0.0  GAUSS:0.1:0.1:0.003
