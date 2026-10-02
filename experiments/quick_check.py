"""One-minute sanity check: levels 0, 1, 2 of the vertex LMIs on the polytope co{A1, A2(a)} for a = 8.
Expected output: level 0 infeasible (a > 3 + 2 sqrt 2), levels 1 and 2 feasible."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "polytope"))
import polytope_levels as pl
for N in (0, 1, 2):
    print("a = 8, level N = %d : feasible = %s" % (N, pl.jet_finsler(8.0, N)))
