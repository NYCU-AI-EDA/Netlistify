
.subckt 000305_output
i0 gnd net3 i
i1 gnd net4 i
r2 net0 net2 r
r3 net0 net1 r
q4 net1 net4 gnd npn
q5 net2 net3 gnd npn
c6 net2 net4 c
c8 net1 net3 c
.ends
