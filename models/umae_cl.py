import copy
import torch
from torch import nn
from transformers import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel, BertForMaskedLM
from torch.nn import CrossEntropyLoss
from timm.models.vision_transformer import Block
from models import models_vit
import timm
from models.data2vec import beit_base_patch16_224
import math
import warnings
import torch.nn.functional as F


BertLayerNorm = torch.nn.LayerNorm
class UmaeCL(nn.Module):
    """For joint visual language embedding"""
    def __init__(self, args, freeze_encoder_flag=False):
        super().__init__()
        self.args = args
        # create mae model
        self.model_mae = self.prepare_mae_model(args.mae_chkpt_dir) # 'multi_modal_mae_vit_base'
        # self.model_mae.cuda(args.gpu)
        if 'base' in args.vit_model:
            self.embed_dim = 768
        else:
            self.embed_dim = 1024

    def freeze_encoders(self):
        print("Disable gradient on encoder")
            # requires_grad = False
        mae_grad_list = [] #"cls_token"
        for name, param in self.model_mae.named_parameters():
            if name not in mae_grad_list:
                param.requires_grad = False
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_mae_model(self, chkpt_dir, ):
        model = models_vit.__dict__[self.args.vit_model](
            drop_path_rate=self.args.drop_path,
            global_pool=self.args.global_pool,
        )
        # load model
        if chkpt_dir is not None:
            checkpoint = torch.load(chkpt_dir, map_location='cpu')
            print("Load pre-trained checkpoint of MAE from: %s" % chkpt_dir)

            if 'pretrain' in chkpt_dir:
                checkpoint_model = checkpoint ['model']
                state_dict = model.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

                # interpolate position embedding
                from utils.util import interpolate_pos_embed
                interpolate_pos_embed(model, checkpoint_model)
                msg = model.load_state_dict(checkpoint['model'], strict=False)
            else:
                msg = model.load_state_dict(checkpoint['model'], strict=False)

            print("message of loading MAE model: \n",msg)

            # if self.args.global_pool:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        return model

    def update_cls_num(self, num_classes):

        self.model_mae.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.model_mae.head.apply(self._init_weights)

    def forward_img_encoder(self, x):
        # x = x.cuda()
        out = self.model_mae.forward_features(x)

        return out

    def forward(self, inputs):
        # imgs = inputs['img']
        img_latent= self.forward_img_encoder(inputs)

        # img_latent = img_latent.unsqueeze(1)

        return img_latent

    def forward_classfication(self, inputs):

        out = self.model_mae.forward(inputs)

        return out

mae_model_dict = {
    'mae_cl_base': [UmaeCL, 768],
    'mae_cl_large': [UmaeCL, 1024],

}

class Vit(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.encoder = timm.create_model(name, pretrained=True)

    def forward(self, inputs):
        return self.encoder.forward_features(inputs)

data2vec_config = { 'num_classes': 1000,
                    'drop_rate': 0,
                    'drop_path_rate': 0.2,
                    'attn_drop_rate': 0,
                    'use_mean_pooling': False,
                    'init_scale': 0.001,
                    'use_rel_pos_bias': False,
                    'use_shared_rel_pos_bias': False ,
                    'use_abs_pos_emb': False,
                    'init_values': 1e-5,
                    'linear_classifier': False,
                    'has_masking': False,
                    'learn_layer_weights': False,
                    'layernorm_before_combine': False,
                  }

class Data2vec(nn.Module):
    def __init__(self, name, chkpt_dir=None):
        super().__init__()
        if name == 'data2vec_base':
            self.encoder = beit_base_patch16_224(**data2vec_config)
        # load model
        if chkpt_dir is not None:
            checkpoint = torch.load(chkpt_dir, map_location='cpu')
            print("Load pre-trained checkpoint of Data2vec from: %s" % chkpt_dir)
            msg = self.encoder.load_state_dict(checkpoint['model'], strict=False)
            print("message of loading Data2vec model: \n", msg)

        if 'base' in name:
            self.embed_dim = 768
        else:
            self.embed_dim = 1024

    def forward(self, inputs):
        return self.encoder.forward_features(inputs)

    def forward_classfication(self, inputs):

        out = self.encoder.forward(inputs)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_cls_num(self, num_classes):

        self.encoder.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.encoder.head.apply(self._init_weights)

data2vec_model_dict = {
    'data2vec_base': [Data2vec, 768],

}

class GeneralImgModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.encoder = timm.create_model(name, pretrained=True)

    def forward(self, inputs):
        return self.encoder.forward(inputs)

from functools import partial

simmim_config = { 'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'num_classes':1000,
        'embed_dim':768,
        'depth':12,
        'num_heads':12,
        'mlp_ratio':4.,
        'qkv_bias':True,
        'drop_rate':0.0,
        'drop_path_rate':0.1,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6),
        'init_values':0.1,
        'use_abs_pos_emb':False,
        'use_rel_pos_bias':True,
        'use_shared_rel_pos_bias':False,
        'use_mean_pooling':True
                  }

from models.simmim import VisionTransformer

def build_simmim(config):
    model = VisionTransformer(**config)

    return model

class Simmim(nn.Module):
    def __init__(self, name, chkpt_dir=None):
        super().__init__()

        if name == 'simmim_base_vit':
            self.encoder = build_simmim(simmim_config)
        print(chkpt_dir)
        # load model
        if chkpt_dir is not None:
            checkpoint = torch.load(chkpt_dir, map_location='cpu')
            print("Load pre-trained checkpoint of Simmim from: %s" % chkpt_dir)
            msg = self.encoder.load_state_dict(checkpoint['model'], strict=False)
            print("message of loading Simmim model: \n", msg)
        else:
            print("do not load simmim weight")

        if 'base' in name:
            self.embed_dim = 768
        # else:
        #     self.embed_dim = 1024

    def forward(self, inputs):
        return self.encoder.forward_features(inputs)

    def forward_classfication(self, inputs):

        out = self.encoder.forward(inputs)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_cls_num(self, num_classes):

        self.encoder.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.encoder.head.apply(self._init_weights)


model_dict = {
    'vit_base_patch16_224': [Vit, 768],
    'data2vec_base': [Data2vec, 768],
    'mae_cl_base': [UmaeCL, 768],
    'simmim_base_vit':[Simmim,768]
}

class LaCViTNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, args, name='', head='mlp', feat_dim=128, project_flag=True):
        super().__init__()
        if 'mae_cl' in name:
            model_fun, dim_in = mae_model_dict[name]
            self.encoder = model_fun(args,)
        elif 'simmim' in name:
            model_fun, dim_in = model_dict[name]
            self.encoder = model_fun(name, args.simmim_chkpt)

        elif 'vit' in name:
            model_fun, dim_in = model_dict[name]
            self.encoder = model_fun(name)
        elif 'data2vec' in name:
            model_fun, dim_in = data2vec_model_dict[name]
            self.encoder = model_fun(name, args.data2vec_chkpt)

        else:
            self.encoder =  GeneralImgModel(name)
            if 'base' in name:
                dim_in = 768
            else:
                raise Exception("Sorry, could not indentify the size of model")

        # self.out_dim = dim_in
        self.project_flag = project_flag
        if self.project_flag == True:
            if head == 'linear':
                self.head = nn.Linear(dim_in, feat_dim)
            elif head == 'mlp':
                self.head = nn.Sequential(
                    nn.Linear(dim_in, dim_in),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_in, feat_dim)
                )
            else:
                raise NotImplementedError(
                    'head not supported: {}'.format(head))

    def forward(self, x):
        feats = self.encoder(x)
        # feats = feats.view(feats.size()[0], -1)
        if self.project_flag == False:
            return feats
        feats = F.normalize(self.head(feats), dim=1)
        return feats

class SupCENet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))

class SupCEMAE(nn.Module):
    """encoder + classifier"""
    def __init__(self, opt, name, num_classes):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        if 'data2vec' in name:
            self.encoder = model_fun(name, opt.data2vec_chkpt)

        elif 'simmim' in name:
            model_fun, dim_in = model_dict[name]
            self.encoder = model_fun(name, opt.simmim_chkpt)

        elif 'vit' in name:
            self.encoder = model_fun(name)
        else:
            self.encoder = model_fun(opt)

        self.encoder.update_cls_num(num_classes)

    def forward(self, x):
        return self.encoder.forward_classfication(x)

    def forward_features(self, x):
        return self.encoder(x)

class ConMAE(nn.Module):
    def __init__(self, args, name='',):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(args,)
        self.out_dim = dim_in

    def forward(self, x):
        feats = self.encoder(x)
        return feats


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

        self.apply(self._init_weights)

    def forward(self, features):
        return self.fc(features)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

