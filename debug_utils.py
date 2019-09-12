import numpy as np
from shapely.geometry import Polygon
import torch
import torch.nn.functional as F
import os
from PIL import Image, ImageOps, ImageStat, ImageDraw
def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def standard_nms(S, thres):
    """ use pre_thres to filter """
    index = np.where(S[:, 8] > thres)[0]
    S = S[index] # ~ 100, 4

    # Then use standard nms
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])
        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]
    return S[keep]


def get_abs_boxes(anchors,network_reg,shift=None):
    tortype= torch.Tensor
    nptype = type(np.zeros(1))
    if type(network_reg) is tortype:network_reg=network_reg.cpu().detach().numpy()
    assert type(network_reg) == nptype
    x = anchors[:,0] + anchors[:, 2] * network_reg[:, 0]
    y = anchors[:,1] + anchors[:, 3] * network_reg[:, 1]
    w = anchors[:,2] * np.exp(network_reg[:, 2])
    h = anchors[:,3] * np.exp(network_reg[:, 3])
    return to_abs_pos(np.stack([x,y,w,h]),shift)


def penalty_score(score,delta,state):
    assert delta.shape[0]==4

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    p=state['p']
    target_sz=state['target_sz']
    scale_z  =state['scale_z']
    s_z=target_sz*scale_z

    s_c = change(sz(delta[2, :], delta[3, :]) / sz_wh(s_z))  # scale penalty
    r_c = change((s_z[0] / s_z[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty
    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score
    return pscore

def window_cosine(pscore,state):
    p=state['p']
    window_influence=state['p'].window_influence
    window    = p.window
    pscore    = pscore * (1 - window_influence) + window * window_influence
    score     = pscore #from 0.2 - 0.7
    return score

def to_abs_pos(rel_pos,shift):
    if shift is not None:
        x,y,w,h=rel_pos
        s_x,s_y,s_w,s_h,lr,resize=shift
        x = s_x + x/resize
        y = s_y + y/resize
        w = s_w * (1 - lr) + w * lr /resize
        h = s_h * (1 - lr) + h * lr /resize
        return np.stack([x,y,w,h])
    else:
        return rel_pos


def proposal_nms(abs_boxes,score,boundary,nms,shift=None,nms_threshold = 0.6,verbose=False):
    def reshape(x):
        t = np.array(x, dtype = np.float32)
        return t.reshape(-1, 1)
    x,y,w,h=to_abs_pos(abs_boxes,shift)
    b_w,b_h=boundary
    x1 = np.clip(x-w//2, 0, b_w)
    x2 = np.clip(x+w//2, 0, b_w)
    x3 = np.clip(x+w//2, 0, b_w)
    x4 = np.clip(x-w//2, 0, b_w)
    y1 = np.clip(y-h//2, 0, b_h)
    y2 = np.clip(y-h//2, 0, b_h)
    y3 = np.clip(y+h//2, 0, b_h)
    y4 = np.clip(y+h//2, 0, b_h)
    slist = map(reshape, [x1, y1, x2, y2, x3, y3, x4, y4, score])
    s = np.hstack(slist)
    maxscore = max(s[:, 8])
    if nms and maxscore > nms_threshold:
        proposals = standard_nms(s, nms_threshold)
        proposals = proposals if proposals.shape[0] != 0 else s
        if verbose:print('nms spend {:.2f}ms'.format(1000*(time.time()-start)))
    else:
        proposals = s
    return proposals

def draw_line(draw,center_x,center_y,width,height,color='white'):
    x1s, y1s, x2s, y2s = center_x - width//2, center_y - height//2, center_x + width//2, center_y + height//2
    try:
        for i in range(len(x1s)):
            x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill=color) # pos anchor
    except:
        draw.line([(x1s, y1s), (x2s, y1s), (x2s, y2s), (x1s, y2s), (x1s, y1s)], width=1, fill=color) # pos anchor

def back2origin_size(one_box_four_corner,origin_w,origin_h,ratio,crop_x,crop_y):
    w,h   =origin_w,origin_h
    x_, y_=crop_x,crop_y
    x1, y1, x3, y3=one_box_four_corner
    """ un resized """
    x1, y1, x3, y3 = x1/ratio, y1/ratio, y3/ratio, y3/ratio
    """ un cropped """
    x1 = np.clip(x_ + x1, 0, w-1).astype(np.int32) # uncropped #target_of_original_img
    y1 = np.clip(y_ + y1, 0, h-1).astype(np.int32)
    x3 = np.clip(x_ + x3, 0, w-1).astype(np.int32)
    y3 = np.clip(y_ + y3, 0, h-1).astype(np.int32)
    return x1, y1, x3, y3

def debug(epoch,example,cout,state,reg_pred,reg_target,pos_index,config):
    tmp_dir     =config.tmp_dir
    draw_picture=config.draw_picture
    draw_field  =config.draw_field
    isnms       =config.isnms
    isb2o       =config.isb2o
    verbose     =config.verbose
    draw_best_n =config.draw_best_n

    score       = F.softmax(cout, dim=0).data[1,:].cpu().numpy()

    p          = state['p']
    scale_z    = state['scale_z']
    w,h        = state['target_sz']
    pos_x,pos_y= state['target_pos']

    anchors  = state['ag'].anchors#一开始的anchors是对检测图像271x271而言,最后要转变到原图
    #anchors  = anchors 

    #得到的是相对的anchor
    pred_abx = get_abs_boxes(anchors,reg_pred)
    real_abx = get_abs_boxes(anchors,reg_target)

    pscore   = penalty_score(score,pred_abx,state)
    score    = window_cosine(pscore,state)
    best_pscore_id = np.argmax(score)
    lr = pscore[best_pscore_id]* p.lr
    shift=[pos_x,pos_y,w,h,lr,scale_z]
    boundary_y,boundary_x,_=state['im'].shape
    proposals= proposal_nms(pred_abx,score,(boundary_x,boundary_y),isnms,shift)

    indexes  = np.argsort(proposals[:, 8])[::-1]

    l_indexes= len(indexes)

    if draw_best_n>l_indexes:draw_best_n=l_indexes
    indexes  = indexes[:draw_best_n]
    x1, y1, x2, y2, x3, y3, x4, y4, _ = proposals[indexes[0]]
    if not draw_picture and verbose:print('no debug image output')
    if draw_picture:
        assert tmp_dir is not None
        if draw_field is [] and verbose:print('only the best image output')
        if not os.path.exists(tmp_dir):os.makedirs(tmp_dir)
        detection   = Image.open(state['im_path'])
        draw        = ImageDraw.Draw(detection)

        if pos_index is not None:
            # draw pos anchors
            if 'positive_anchors' in draw_field:
                rel_pos = anchors[pos_index].transpose()
                shift=[pos_x,pos_y,w,h,1,scale_z]
                x,y,w,h = to_abs_pos(rel_pos,shift)
                draw_line(draw,x,y,w,h,color="white")

#             #pos anchor transform to red box after prediction
#             if 'pred_pos_anchor' in draw_field:
#                 rel_pos = real_abx[:,pos_index]
#                 shift=[pos_x,pos_y,w,h,1,scale_z]
#                 x,y,w,h = to_abs_pos(rel_pos,shift)
#                 draw_line(draw,x,y,w,h,color="red")

            #groud truth
            if 'groud_truth' in draw_field:
                x,y=state['target_pos']
                w,h=state['target_sz']
                draw_line(draw,x+2,y+2,w,h,color="green")
                #red should be same as green
                rel_pos = real_abx[:,pos_index]
                shift=[pos_x,pos_y,w,h,1,scale_z]
                x,y,w,h = to_abs_pos(rel_pos,shift)
                draw_line(draw,x,y,w,h,color="red")


        for i in range(draw_best_n):
            x1, y1, x2, y2, x3, y3, x4, y4, _ = proposals[indexes[i]]
            draw.line([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)], width=3, fill='yellow')

        save_path = os.path.join(tmp_dir, 'epoch_{:010d}_{:010d}_anchor_pred.jpg'.format(epoch, example))
        detection.save(save_path)
