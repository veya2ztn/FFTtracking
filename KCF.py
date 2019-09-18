import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.autograd import Variable
import cv2
from model_utils import *

def KCF(xes,yes,zes,tes,imgp):
    self_entangle = entangle_gauss(xes,xes)
    cros_entangle = entangle_gauss(zes,xes)

    final_phase_y = fullfft(yes)

    B,C,W,H = self_entangle.shape
    final_phase_x = fullfft(self_entangle)
    final_phase_z = cros_entangle

    alphaf = complex_div(final_phase_y,final_phase_x+0.001)

    alpha  = torch.irfft(alphaf,2,onesided=False)
    final_phase_o=entangle_tensor(final_phase_z,alpha,'fft','nn',sum_channel=1)
    response=final_phase_o[0,0].cpu().numpy()
    response-= response.min()
    response/= response.max()
    v_centre, h_centre = np.unravel_index(response.argmax(), response.shape)
    vert_delta, horiz_delta = [v_centre - response.shape[0] / 2,
                               h_centre - response.shape[1] / 2]
    vert_delta /=imgp.scale_z
    horiz_delta/=imgp.scale_z
    t=tes[0,0].cpu().numpy()
    y=yes[0,0].cpu().numpy()
    r=response
    a=cv2.resize(imgp.template_patch_win,(256,256))
    b=imgp.detected_patch
    c=cv2.merge([t,t,t])
    d=cv2.merge([r,r,r])
    y=cv2.merge([y,y,y])
    y=cv2.resize(y,(256,256))
    temp1= np.concatenate([a,b,b],1)
    temp2= np.concatenate([y,c,d],1)
    return np.concatenate([temp1,temp2])
