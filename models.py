from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from backbone import BodyLayer

from model_utils import *

class TransposeBottleneckV1(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, inplanes, output_channels, stride=1, norm_layer=nn.BatchNorm2d):
        super(TransposeBottleneckV1, self).__init__()
        planes=(output_channels+inplanes)//2
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = norm_layer(planes)
        optpad = stride-1
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride,padding=1,output_padding=optpad, bias=False)
        self.bn2   = norm_layer(planes)
        self.conv3 = nn.ConvTranspose2d(planes, output_channels, kernel_size=1, bias=False)
        self.bn3   = norm_layer(output_channels)
        self.relu  = nn.ReLU(inplace=True)
        #self.relu  = nn.LeakyReLU(negative_slope=0.6, inplace=True)
        self.stride = stride
        self.upsample =None
        if output_channels!=inplanes or stride >1:
            self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(inplanes,
                                       output_channels,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       output_padding=optpad,
                                       bias=False) ,
                    norm_layer(output_channels),
                )
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class PositiveBottleneckV1(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, inplanes, output_channels, stride=1, norm_layer=nn.BatchNorm2d, dilation=1):
        super(PositiveBottleneckV1, self).__init__()
        planes=(output_channels+inplanes)//2
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dilation=dilation)
        self.bn1   = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=dilation, bias=False, dilation=dilation)
        self.bn2   = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, output_channels, kernel_size=1, bias=False, dilation=dilation)
        self.bn3   = norm_layer(output_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if output_channels!=inplanes or stride >1:
            self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes,
                              output_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False) ,
                    norm_layer(output_channels),
                )
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TransposeBottleneckV2(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, inplanes, output_channels, stride=1, norm_layer=nn.BatchNorm2d):
        super(TransposeBottleneckV3, self).__init__()
        planes=(output_channels+inplanes)//2
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = norm_layer(planes)
        optpad = stride-1
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride,padding=1,output_padding=optpad, bias=False)
        self.bn2   = norm_layer(planes)
        self.conv3 = nn.ConvTranspose2d(planes, output_channels, kernel_size=1, bias=False)
        self.bn3   = norm_layer(output_channels)
        #self.relu  = nn.ReLU(inplace=True)
        self.relu  = nn.LeakyReLU(negative_slope=0.6, inplace=True)
        self.stride = stride
        self.upsample =None

    def forward(self, x):
        #residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)

        return out

class UPDownLayer(nn.Module):
    def __init__(self,inchan,outchan,resize,layers_num=3,block=TransposeBottleneckV2,channels_mode='linear'):
        '''
        input_size = (B, inchan,w,h)
        output_size= (B,outchan,W,H)
        resize: the output size W,H equal resize*(w,h)
        layers_num: if the resize set 1 which mean no scalar for size.
                Using [layers] control how many layers in the Network
        block : the CNN-resnet blocks
        channels_mode: 'last' or 'linear' means the channel distribution for Layers
                'last' mean use the last layer up channels
                'linear' mean the channel grow step by step per layer
        '''
        super(UPDownLayer, self).__init__()
        ### require for how many layer totally
        # resize must be 2**n
        order = np.log2(resize)
        assert order == np.floor(order)
        order  = int(order)
        inchan = int(inchan)
        outchan= int(outchan)
        total_layer_num=layers_num
        scalar_list = [1]*total_layer_num
        for i in range(order):
            scalar_list[total_layer_num//(order+1)*i]=2

        if channels_mode == 'last':
            channel_list = [inchan]*(total_layer_num)+[outchan]
        elif channels_mode == 'linear':
            channel_list = np.floor(np.linspace(inchan,outchan,total_layer_num+1))

        self.block=block
        net = nn.Sequential()

        for i in range(total_layer_num):
            ic = int(channel_list[i])
            oc = int(channel_list[i+1])
            scalar = int(scalar_list[i])
            name = str(i)+'_X'+str(scalar)
            net.add_module(name,self.block(ic,oc,scalar))

        self.net=net

    def forward(self,x):
        return self.net(x)

class KSLayer(nn.Module):
    def __init__(self,outchan,inchannel_per_layer,resize_per_layer,block):
        super(KSLayer,self).__init__()
        assert len(inchannel_per_layer) == len(resize_per_layer)
        self.modulelist= nn.ModuleList()
        for inchan,resize in zip(inchannel_per_layer,resize_per_layer):
            self.modulelist.append(UPLayer(inchan,outchan,resize,block))
    def forward(self,_input):
        return [layer(x) for x,layer in zip(_input,self.modulelist)]

class CNNF(nn.Module):

    def __init__(self,cfg):
        super(CNNF,self).__init__()
        backbone_config= cfg.backbone
        fpn_config=cfg.fpn
        self.Body_Layer       = BodyLayer(backbone_config,fpn_config)
        if backbone_config is not None:
            self.entangle         = entangle_list
            self.final_space      = fullfft
            self.tracking_proposal= Naive_KCF
            inchannel_per_layer   = self.Body_Layer.out_channels
            resize_per_layer      = [8,16,32]
        else:
            self.entangle         = entangle_gauss_list
            self.final_space      = fullfft
            inchannel_per_layer   = [1]
            resize_per_layer      = [1]

        KSL_block = TransposeBottleneckV1
        self.Kernel_Simula_Layer = KSLayer(1,inchannel_per_layer,resize_per_layer,KSL_block)
        self.debug = False
        self.mseloss=nn.MSELoss()
        self.KLloss =nn.KLDivLoss(reduction='mean')
        # stratage 1:
        #    compute Y/X in freq space so it is element-wise complex divide
        #    compute (Y/X)*Z in real space so it is conv2 (which is the entangle)
        # stratage 2:
        #    compute Y/X*Z in freq space so it is element-wise complex divide and mul
        #    but now the dimension difficuly comes
        # stratage 3:
        #    Since the target is Real Number in both real and freq space
        #    compute Y/X*Z in freq space so it is element-wise complex divide and mul
        #    but now the dimension difficuly comes
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)


    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

        self.load_state_dict(state_dict, strict=False)

    def forward(self, _input,target=None):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        # Initialize this here or DataParellel will murder me in my sleep
        # which, admittedly, I could use some of right now.
        detected_img,template_img,template_y = _input

        detected_feas  = self.Body_Layer(detected_img)
        template_feas  = self.Body_Layer(template_img)

        self_entanglen = self.entangle(template_feas,template_feas)#B,3,小,小
        cros_entanglen = self.entangle(detected_feas,template_feas)#B,3,大,大

        self_entangles = self.Kernel_Simula_Layer(self_entanglen)#B,1,小,小
        cros_entangles = self.Kernel_Simula_Layer(cros_entanglen)#B,1,大,大

        if self.debug:
            self.debug_detected_feas  = detected_feas
            self.debug_template_feas  = template_feas
            self.debug_self_entanglen = self_entanglen
            self.debug_cros_entanglen = cros_entanglen
            self.debug_self_entangles = self_entangles
            self.debug_cros_entangles = cros_entangles
        out = []
        _,_,dim1,dim2=detected_img.shape
        for self_entangle,cros_entangle in zip(self_entangles,cros_entangles):
        # if the img_process convert the y into fourier phase
        # we know if we set y,t as a normal distribution
        # it's fourier is also normal distribution and only get real part
        # at stratage 1:
            final_phase_y = self.final_space(template_y)
            final_phase_x = self.final_space(self_entangle)
            final_phase_z = cros_entangle
            _lambda = 1
            final_phase_o = torch.irfft(complex_div(final_phase_y,final_phase_x+_lambda),2,onesided=False)
            final_phase_o = entangle_tensor(final_phase_z,final_phase_o,'fft','nn')
            _min = final_phase_o.min()
            _max = final_phase_o.max()
            final_phase_o = (final_phase_o-_min)/(_max-_min)
            # 原则上输出的结果就是 t 或者 F(t), 所以不应该有后续操作
            # final_phase_o = nn.Softmax(2)(final_phase_o.view(-1,dim1,dim2)).view_as(final_phase_o)

            if target is None:
                out.append(final_phase_o)
            else:
                final_phase_t = target

                #loss = self.KLloss(final_phase_o.log(),final_phase_t,)
                loss = self.mseloss(final_phase_o ,final_phase_t)
                out.append(loss)

        return out


        #     final_phase_y = self.final_space(template_y,output_shape=(dim1,dim2))
        #     final_phase_x = self.final_space(self_entangle,output_shape=(dim1,dim2))
        #     final_phase_z = self.final_space(cros_entangle,output_shape=(dim1,dim2))
        #     final_phase_o = self.tracking_proposal(final_phase_y,final_phase_x,final_phase_z)
        #     if target is None:
        #         out.append(final_phase_o)
        #     else:
        #         final_phase_t = self.final_space(target)
        #         loss = self.mseloss(final_phase_t,final_phase_o)
        #         out.append(loss)
        #
        # return out
