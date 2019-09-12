import torch
import numpy as np

def fftshift(tensor:torch.Tensor)-> torch.Tensor:
    '''
    shift the origin to corner
    '''
    assert tensor.shape[-1]==2
    dim1,dim2= tensor.shape[-3:-1]
    tensor=torch.roll(tensor,dim1//2,2)
    tensor=torch.roll(tensor,dim2//2,3)
    return tensor


def ifftshift(tensor:torch.Tensor)-> torch.Tensor:
    assert tensor.shape[-1]==2
    dim1,dim2= tensor.shape[-3:-1]
    tensor=torch.roll(tensor,-dim2//2,3)
    tensor=torch.roll(tensor,-dim1//2,2)
    return tensor

def fftslice(dim1,dim2):
    ind1 = (dim1//2+1)
    ind2 = (dim2//2+1)
    x = torch.arange(ind1-1)
    y = torch.arange(dim2)
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_x_buck = grid_x.flatten()
    grid_y_buck = grid_y.flatten()

    x = torch.tensor([ind1-1])
    y = torch.arange(ind2)
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_x_line = grid_x.flatten()
    grid_y_line = grid_y.flatten()

    grid_x=torch.cat([grid_x_buck,grid_x_line])
    grid_y=torch.cat([grid_y_buck,grid_y_line])

    return grid_x,grid_y

def fftpress(tensor:torch.Tensor,mode='gen',provided=None)-> torch.Tensor:
    '''
    fftshift to the center and get the 1/4 corner
    '''
    assert tensor.shape[-1]==2
    tensor = ifftshift(tensor)
    if mode == 'gen':
        dim1,dim2= tensor.shape[-3:-1]
        a,b=fftslice(dim1,dim2)
    elif mode == 'provide':
        a,b = provided
    return tensor[:,:,a,b,:]

def get_ref_num(num,dim):
    dim1,dim2=dim
    center = (dim1//2,dim2//2)
    coor=(num//dim2,num%dim2)
    refcoor=(2*center[0]-coor[0],2*center[1]-coor[1])
    refnum = dim2*refcoor[0]+refcoor[1]
    return refnum

def ifftslice(dim1,dim2):
    center = (dim1//2,dim2//2)
    cennum = dim2*center[0]+center[1]
    indexs = []
    for i in range(cennum+1,dim1*dim2):
        indexs.append(get_ref_num(i,(dim1,dim2)))
    return torch.LongTensor(indexs)

def ifftpress(tensor:torch.Tensor,origin_dim,mode='gen')-> torch.Tensor:
    '''
    fftshift to the center and get the 1/4 corner
    '''
    assert tensor.shape[-1]==2
    if mode == 'gen':
        dim1,dim2 = origin_dim
        sli = ifftslice(dim1,dim2)
    elif mode =='provide':
        dim1,dim2,sli = origin_dim
        real= tensor[:,:,sli,0]
        imag=-tensor[:,:,sli,1]#conj
        left_part=torch.stack([real,imag],dim=-1)
        head_part=tensor
        origin = torch.cat([head_part,left_part],dim=-2)
        head_shape = list(origin.shape[:-2])
        out_shape  = head_shape +[dim1,dim2,2]
        return origin.reshape(out_shape)
def nprint(tensor):
    nx= tensor.numpy()
    nx= nx[...,0]+1j*nx[...,1]
    print(nx)


def complex_mul(tensor_1: torch.Tensor,tensor_2: torch.Tensor)-> torch.Tensor:
    '''
    :param tensor_1(2) [...,2] for real part and image part
    '''
    assert tensor_1.shape[-1]==2
    assert tensor_2.shape[-1]==2
    real1,imag1=tensor_1[...,0],tensor_1[...,1]
    real2,imag2=tensor_2[...,0],tensor_2[...,1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

def complex_rec(tensor_2: torch.Tensor)-> torch.Tensor:
    assert tensor_2.shape[-1]==2
    c,d=tensor_2[...,0],tensor_2[...,1]
    Denominator = c**2+d**2+10e-10
    return torch.stack([c/Denominator, -d/Denominator], dim = -1)

def complex_div(tensor_1: torch.Tensor,tensor_2: torch.Tensor)-> torch.Tensor:
    assert tensor_1.shape[-1]==2
    assert tensor_2.shape[-1]==2
    assert tensor_1.shape[:-1] == tensor_2.shape[:-1]
    a,b=tensor_1[...,0],tensor_1[...,1]
    c,d=tensor_2[...,0],tensor_2[...,1]
    Denominator = c**2+d**2+0.00000001
    return torch.stack([(a * c + b * d)/Denominator, (b*c-a*d)/Denominator], dim = -1)

def complex_div(tensor_1: torch.Tensor,tensor_2: torch.Tensor)-> torch.Tensor:
    assert tensor_1.shape[-1]==2
    assert tensor_2.shape[-1]==2

    a,b=tensor_1[...,0],tensor_1[...,1]
    c,d=tensor_2[...,0],tensor_2[...,1]
    Denominator = c**2+d**2+0.00000001
    return torch.stack([(a * c + b * d)/Denominator, (b*c-a*d)/Denominator], dim = -1)

def complex_rec(tensor_2: torch.Tensor)-> torch.Tensor:
    assert tensor_2.shape[-1]==2
    c,d=tensor_2[...,0],tensor_2[...,1]
    Denominator = c**2+d**2
    return torch.stack([c/Denominator, -d/Denominator], dim = -1)

def complex_conj(tensor_1: torch.Tensor)-> torch.Tensor:
    assert tensor_1.shape[-1]==2
    real1,imag1=tensor_1[...,0],tensor_1[...,1]
    imag1=-imag1
    return torch.stack([real1,imag1], dim = -1)

def complex_polar(tensor: torch.Tensor)-> torch.Tensor:
    assert tensor.shape[-1]==2
    real,imag=tensor[...,0],tensor[...,1]
    radius = torch.norm(tensor,dim=-1)
    angles = torch.atan(real/imag)
    return torch.stack([radius,angles],dim=-1)

def complex_polar_ln(tensor: torch.Tensor):
    assert tensor.shape[-1]==2
    real,imag=tensor[...,0],tensor[...,1]
    radius = torch.norm(tensor,dim=-1).log()
    angles = torch.atan(real/imag)
    return radius,angles

def cplx_tch2np(tch):
    assert tch.shape[-1]==2
    out=tch.numpy()
    return out[...,0]+1j*out[...,1]
def cplx_np2tch(npx):
    real = torch.Tensor(np.real(npx))
    imag = torch.Tensor(np.imag(npx))
    return torch.cat([real,imag],dim=-1)

def get_norm_per_batch(tensor:torch.Tensor):
    assert len(tensor.shape)==4
    nB=tensor.shape[0]
    tensor=tensor.reshape(nB,-1)
    return (tensor**2).sum(1).reshape(nB,1,1,1)

def entangle_tensor(feature_input,feature_given,method,mode="nn",sum_channel=-1):
    '''
    fft faster for almost case
    test for big =1000,small =20
    '''
    assert len(feature_input.shape)==4
    assert len(feature_given.shape)==4

    _,c1,big,big = feature_input.shape
    _,c2,small,small = feature_given.shape
    assert big >= small
    if method == 'conv':
        #only support 1 channel
        assert c1==1
        assert c2==1
        # use conv achieve F(F(x).*F(z))
        padnum  = big-small
        p1,p2=feature_given.shape[-2:]
        feature_input_p = pad_circular_standard(feature_input,(p1-1,p2-1),mode)
        feature_given_f = torch.flip(feature_given,(2,3))
        out=torch.nn.functional.conv2d(feature_input_p,feature_given_f)
        # use conv achieve F(F(x)* .* F(z))
        if mode == "*n":out = torch.flip(out,(-2,-1))
        return out
    elif method == 'fft':
        # use fft achieve F(F(x).*F(z))
        padnum  = big-small
        feature_given_p = torch.nn.functional.pad(feature_given,(0,padnum,0,padnum))
        f_input = torch.rfft(feature_input,2,onesided=False)
        f_given = torch.rfft(feature_given_p,2,onesided=False)
        # use fft achieve F(F(x)* .* F(z))
        if mode == "*n":f_given=complex_conj(f_given)
        f_tangl = complex_mul(f_input,f_given)
        if sum_channel >=0:
            # this case for multiply channels image
            # the fft for (B,C,W,H) is (B,C,W,H,2)
            # So we add all channels value together.
            f_tangl = f_tangl.sum(dim=sum_channel,keepdim=True)
        out     =torch.irfft(f_tangl,2,onesided=False)
        return out
    else:
        raise NotImplementedError

def entangle_gauss(feature_input,feature_given,sigma=0.2):
    # feature_input is big one
    N  = feature_input.shape[-1] * feature_input.shape[-2]
    NI = get_norm_per_batch(feature_input)
    NG = get_norm_per_batch(feature_given)
    cc = entangle_tensor(feature_input,feature_given,method='fft',mode="*n",sum_channel=1)
    d = NI + NG - 2 * cc
    kk = torch.exp(-1. / 0.2 ** 2 * torch.abs(d) / N)
    return kk



def entangle_xz(feature_input,feature_given):
    #tensor rule (B,C,W,H)
    B,C,W,H=feature_input.shape
    entangle = entangle_tensor(feature_input,feature_given,method='fft',mode="*n")
    NI = (feature_input**2).mean()*torch.ones(B,1,W,H,device=feature_input.device)
    NG = (feature_given**2).mean()*torch.ones(B,1,W,H,device=feature_input.device)
    out= torch.cat([NI,entangle,NG],dim=1)
    return out

def entangle_gauss_list(feature_inputs,feature_givens,sigma=0.2):
    return [entangle_gauss(feature_input,feature_given,sigma) for feature_input,feature_given in zip(feature_inputs,feature_givens)]

def entangle_list(feature_inputs,feature_givens,method,mode="nn"):
    '''
    fft faster for almost case
    test for big =1000,small =20
    '''
    return [entangle_tensor(feature_input,feature_given,method,mode) for feature_input,feature_given in zip(feature_inputs,feature_givens)]

def entangle_freq(feature_input,feature_given,mode="*n"):
    '''
    fft faster for almost case
    test for big =1000,small =20
    '''
    assert len(feature_input.shape)==4
    assert len(feature_given.shape)==4
    _,_,big,big = feature_input.shape
    _,_,small,small = feature_given.shape
    assert big >= small
    padnum  = big-small
    feature_given_p = torch.nn.functional.pad(feature_given,(0,padnum,padnum,0))
    f_input = torch.rfft(feature_input,2,onesided=False)
    f_given = torch.rfft(feature_given_p,2,onesided=False)
    if mode == "*n":f_given=complex_conj(f_given)
    f_tangl = complex_mul(f_input,f_given)
    return f_tangl

def pad_circular(x, pad,mode):
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :return:
    """
    assert len(x.shape)==2
    pad1,pad2=pad
    if mode=="nn":
        x = torch.cat([x[:, -pad1:],x], dim=1)
        x = torch.cat([x[-pad2:], x], dim=0)
        return x
    elif mode=="*n":
        x = torch.flip(x,(0,1))
#         x = torch.cat([x, x[0:pad1]], dim=0)
#         x = torch.cat([x, x[:, 0:pad2]], dim=1)
        x = torch.cat([x[:, -pad1:],x], dim=1)
        x = torch.cat([x[-pad2:], x], dim=0)
        return x

def pad_circular_standard(x,pad,mode):

    nB,nC,W,H=x.shape
    pad1,pad2=pad
    img_list=x.reshape(-1,W,H)
    out=torch.Tensor()
    if x.is_cuda:out=out.cuda()

    for img in img_list:
        out=torch.cat([out,pad_circular(img,(pad1,pad2),mode=mode)])
    return out.reshape(nB,nC,W+pad1,H+pad2)

def fullfft(x,output_shape=None):
    if output_shape is None:
        if isinstance(x,list):
            return [torch.rfft(y,2,onesided=False) for y in x]
        return torch.rfft(x,2,onesided=False)
    else:
        dim1,dim2 = output_shape
        if isinstance(x,list):
            outs=[]
            for xx in x:
                _,_,xd1,xd2=xx.shape
                pd1 = dim1-xd1
                pd2 = dim2-xd2
                out = torch.nn.functional.pad(xx,(0,pd1,pd2,0))
                out = torch.rfft(out,2,onesided=False)
                outs.append(out)
            return outs
        else:
            xx=x
            _,_,xd1,xd2=xx.shape
            pd1 = dim1-xd1
            pd2 = dim2-xd2
            out = torch.nn.functional.pad(xx,(0,pd1,pd2,0))
            out = torch.rfft(out,2,onesided=False)
            return out
import torch.nn as nn
# recommend
def zero_weights_init(m):

    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight,0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.constant_(m.weight,0)

def KCF_Core(y,x,z,_lambda=1):
    o = complex_mul(y,z)
    o = complex_div(o,x+_lambda)
    return o

def Naive_KCF(y,x,z,_lambda=1):
    if isinstance(x,list) and isinstance(z,list) and isinstance(y,list):
        assert len(x)==len(z)==len(y)
        return [KCF_Core(yy,xx,zz,_lambda) for yy,xx,zz in zip(y,x,z)]
    if isinstance(x,list) and isinstance(z,list):
        assert len(x)==len(z)
        return [KCF_Core(y,xx,zz,_lambda) for xx,zz in zip(x,z)]
    if isinstance(x,list):
        return [KCF_Core(y,xx,z,_lambda) for xx in x]
    if isinstance(z,list):
        return [KCF_Core(y,x,zz,_lambda) for zz in z]
    return KCF_Core(y,x,z,_lambda)
