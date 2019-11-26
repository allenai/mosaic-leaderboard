import random

for i in range(1767):
    preds = []
    for i in range(5):
        if random.random()>0.5:
            preds.append('1')
        else:
            preds.append('2')
    print(",".join(preds))

    
