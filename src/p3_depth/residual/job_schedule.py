"""Deterministic epoch or fixed-update schedules, including exact exposure counts."""
import math
import numpy as np


def segments(fits, seed, training):
    size=training['batch_size']
    max_steps=training.get('max_steps')
    if not max_steps:
        for epoch in range(training['max_epochs']):
            order=np.random.default_rng(seed+epoch).permutation(fits).tolist()
            yield epoch+1,[order[i:i+size] for i in range(0,len(order),size)]
        return
    # Full batches across permutation boundaries: identical steps and image
    # exposures for every label-set size, with no dropped examples.
    epoch=0;pending=[];batches=[];segment=0
    for step in range(max_steps):
        while len(pending)<size:
            pending.extend(np.random.default_rng(seed+epoch).permutation(fits).tolist())
            epoch+=1
        batches.append(pending[:size]);pending=pending[size:]
        if (step+1)%training.get('evaluate_every_steps',200)==0 or step+1==max_steps:
            segment+=1
            yield segment,batches
            batches=[]


def schedule_steps(fits,training):
    return training.get('max_steps') or training.get('schedule_epochs',training['max_epochs'])*math.ceil(len(fits)/training['batch_size'])
