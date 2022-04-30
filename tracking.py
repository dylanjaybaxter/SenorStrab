# Tracking Functions
import numpy as np
import cv2 as cv

class tracking_obj:
    def __init__(self, mask = None, label = None, vel=0):
        self.mask = mask
        self.label = label
        self.vel = vel

def relabel(obj, obj_prev, max_label):
    obj_new = [None]*len(obj)
    mx_lab = max_label
    # Previous is none
    if obj_prev is None:
        return obj, len(obj)
    else:
        for i in range(0,obj_size(obj)):
            matched = False
            for j in range(0,obj_size(obj_prev)):
                intersect = cv.bitwise_and(obj[i].mask, obj_prev[j].mask)
                sze = sum(sum(obj[i].mask))
                sze2 = sum(sum(obj_prev[j].mask))
                sze_int = sum(sum(intersect))
                if sze_int > 0.25*max(sze,sze2):
                    obj_new[i] = tracking_obj(obj[i].mask,obj_prev[j].label)
                    matched = True;
            if matched == False:
                max_label = max_label + 1
                obj_new[i] = tracking_obj(obj[i].mask, max_label)
        return obj_new, max_label


def obj_size(obj):
    n=0
    while(1):
        if obj[n] is None:
            return n
        else:
            n= n+1