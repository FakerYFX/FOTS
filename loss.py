import torch
import torch.nn as nn
import config as cfg
from torch.autograd import Variable


def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x

def reduce_min(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.min(a, keepdim=keepdim)[0]
    return x


def reduce_max(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.max(a, keepdim=keepdim)[0]
    return x


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (t.log((1 + x) / (1 - x))) * 0.5


def l2r_dist(x, y, keepdim=True, eps=1e-8):
    d = (x - y)**2
    d = reduce_sum(d, keepdim=keepdim)
    d += eps  # to prevent infinite gradient at 0
    return d.sqrt()


def l2_dist(x, y, keepdim=True):
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)


def l1_dist(x, y, keepdim=True):
    d = t.abs(x - y)
    return reduce_sum(d, keepdim=keepdim)


def l2_norm(x, keepdim=True):
    norm = reduce_sum(x*x, keepdim=keepdim)
    return norm.sqrt()


def l1_norm(x, keepdim=True):
    return reduce_sum(x.abs(), keepdim=keepdim)


def rescale(x, x_min=-1., x_max=1.):
    return x * (x_max - x_min) + x_min


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (t.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

# def cross_quad_loss(predict,label,cfg:cfg.DefaultConfig):
#
#     # loss for inside_score
#     logits=predict[:,:,:,:1]
#     labels=label[:,:,:,:1]
#     #balance positive and negative samples in an image
#     beta=1-t.mean(labels)
#     #first appl sigmoid activation
#     predicts=t.sigmoid(logits)
#
#     inside_score_loss=t.mean(-1*(beta*labels*p.log(predicts+cfg.epsilon)+(1-beta)*(1-labels+cfg.epsilon)))
#
#     inside_score_loss*=cfg.lambda_inside_score_loss
#
#     # loss for side_vertex_code
#     vertex_logits=predict[:,:,:,1:3]
#     vertex_labels=label[:,:,:,1:3]
#     vertex_beta=1-(t.mean(predict[:,:,:,1:2])/t.mean(labels)+cfg.epsilon)
#
#     vertex_predicts=t.sigmoid(vertex_logits)
#     pos=-1*vertex_beta*vertex_labels*t.log(vertex_predicts+cfg.epsilon)
#     neg=-1*(1-vertex_beta)*(1-vertex_labels)*t.log(1-vertex_predicts+cfg.epsilon)
#
#     positive_weights=predict[:,:,:,0].eq(1).float()
#
#
#     side_vertex_code_loss=reduce_sum(reduce_sum(pos+neg),axis=-1)*positive_weights/(reduce_sum(positive_weights)+cfg.epsilon)
#     side_vertex_code_loss*=cfg.lambda_side_vertex_code_loss
#
#     #loss for side_vertex_coord delta
#     g_hat=predict[:,:,:,3:]
#     g_label=label[:,:,:,3:]
#
#     vertex_weights=predict[:,:,:,1].eq(1).float()
#
#     smooth_l1_loss_fn=t.nn.SmoothL1Loss()
#     pixel_wise_smooth_l1norm=smooth_l1_loss_fn(g_hat,g_label)
#
#     side_vertex_coord_loss=reduce_sum(pixel_wise_smooth_l1norm)/(reduce_sum(vertex_weights)+cfg.epsilon)
#
#     side_vertex_coord_loss*=cfg.lambda_side_vertex_coord_loss
#
#     return inside_score_loss+side_vertex_code_loss+side_vertex_coord_loss


#add(by xyf)
def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)

    return loss

class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()
        return

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                training_mask):
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        # d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        #d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - torch.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta

        return torch.mean(L_g * y_true_cls * training_mask) + classification_loss

