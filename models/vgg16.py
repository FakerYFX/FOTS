import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torchvision



model_url ='https://download.pytorch.org/models/vgg16-397923af.pth',


def VGG16(pretrained=True):
    """VGG is the vgg net
    
    
    
    
    """
    model =torchvision.models.vgg16(pretrained=pretrained)
    if  pretrained:
        return model
    model.load_state_dict(model_zoo.load_url(model_url))
    return model


