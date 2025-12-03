
.subckt 000299_output
v0 net4 gnd v
r1 net1 net3 r
i2 net5 gnd i
r3 net0 net2 r
i4 net6 gnd i
q5 net2 net4 net5 npn
q6 net3 net2 net6 npn
c9 net5 net6 c
.ends
