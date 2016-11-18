import numpy as np

genes = [line.rstrip('\n') for line in open('stage2_ga2')]
new = []

for i in range(len(genes)):
    if genes[i] not in new:
        new.append(genes[i])
    else:
        print "Duplicate removed:", genes[i]

np.savetxt('stage2_ga2', new, delimiter='\n', fmt='%s')