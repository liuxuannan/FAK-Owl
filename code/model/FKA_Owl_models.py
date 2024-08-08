import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from timm.models.layers import trunc_normal_



class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)

    
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i].transpose(0,1)
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens


class Cross_Modal_Reason(nn.Module):
    def __init__(self, embeding):
        super().__init__()

        self.norm_layer_vision_guided = nn.LayerNorm(embeding)
        self.vision_guided_attention = nn.MultiheadAttention(embeding, 16, dropout=0.0, batch_first=True)

        self.norm_layer_text_guided = nn.LayerNorm(embeding)
        self.text_guided_attention = nn.MultiheadAttention(embeding, 16, dropout=0.0, batch_first=True)

        self.forgery_proj = nn.Linear(2048, 4096)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def forward(self, bs, img_embed = None, text_embed = None):
        if img_embed is not None and text_embed is not None:
            vision_cross_forgery_tokens = \
                self.vision_guided_attention(query=self.norm_layer_vision_guided(img_embed),
                                             key=self.norm_layer_vision_guided(text_embed),
                                             value=self.norm_layer_vision_guided(text_embed))[0]

            text_cross_forgery_tokens = \
                self.text_guided_attention(query=self.norm_layer_text_guided(text_embed),
                                             key=self.norm_layer_text_guided(img_embed),
                                             value=self.norm_layer_text_guided(img_embed))[0]

            vision_forgery_tokens = vision_cross_forgery_tokens[:, 0, :].unsqueeze(1)
            text_forgery_tokens = text_cross_forgery_tokens[:,0,:].unsqueeze(1)

            multimodal_forgery_tokens = torch.cat([vision_forgery_tokens,text_forgery_tokens],dim=2)

            forgery_embed = self.forgery_proj(multimodal_forgery_tokens)

            return forgery_embed,  vision_cross_forgery_tokens[:, 1:, :]

        else:
            base_prompts = self.base_prompts.expand(bs, -1, -1)

            return base_prompts

class Norm2d(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps
        self.normalized_shape = (embed_dim,)

        #  >>> workaround for compatability
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln.weight = self.weight
        self.ln.bias = self.bias

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Segmentation_Verification(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, dim_out, kernel_size=5, padding=0),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU(inplace=True),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, dim_out)),requires_grad=True)

        self.mask_map = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            Norm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.img_proj = nn.Linear(1024, 256)
        self.text_proj = nn.Linear(1024, 256)

    def forward(self, cross_patch_token, feats_text_tensor, atts_cls_feat = None, atts_bbox_feat = None):
        img_patch_token = self.img_proj(cross_patch_token)
        text_feat_token = self.text_proj(feats_text_tensor)

        B, L, C = img_patch_token.shape
        H = int(np.sqrt(L))
        forgery_map_low_level = img_patch_token.permute(0, 2, 1).view(B, C, H, H)
        forgery_map_low_level = self.mask_map(forgery_map_low_level)
        forgery_map_embed = forgery_map_low_level.view(B,C,-1).permute(0,2,1)

        forgery_map_embed = forgery_map_embed / forgery_map_embed.norm(dim=-1, keepdim=True)
        forgery_map_high_level = (100.0 * forgery_map_embed @ text_feat_token.transpose(-2, -1))
        B, L, C = forgery_map_high_level.shape
        H = int(np.sqrt(L))
        forgery_maps = F.interpolate(forgery_map_high_level.permute(0, 2, 1).view(B, 2, H, H),
                                    size=224, mode='bilinear', align_corners=True)
        forgery_maps = torch.softmax(forgery_maps, dim=1)
        forgery_map = forgery_maps[:,1,:,:].unsqueeze(1)

        B,C,H,W = forgery_map.shape
        img_prompts = self.meta_net(forgery_map)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,4096,9).transpose(-2,-1)
        if atts_cls_feat != None:
            base_prompts = self.base_prompts.expand(B, -1, -1) + atts_cls_feat
        else:
            base_prompts = self.base_prompts.expand(B, -1, -1)

        if atts_bbox_feat != None:
            img_prompts = img_prompts + atts_bbox_feat
        else:
            img_prompts = img_prompts

        output = torch.cat([base_prompts, img_prompts], dim=1)

        return output, forgery_maps


class cls_proj(nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.cls_proj = nn.Linear(input_dim, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def forward(self, cls_embed):

        atts_feat_aggr = self.cls_proj(cls_embed)
        return atts_feat_aggr

class Bbox_Verification(nn.Module):
    def __init__(self, embeding):
        super().__init__()
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, embeding),requires_grad=True)
        self.norm_layer_aggr = nn.LayerNorm(embeding)
        self.aggregator = nn.MultiheadAttention(embeding, 16, dropout=0.0, batch_first=True)

        self.bbox_head = self.build_mlp(input_dim=embeding, output_dim=4)
        self.bbox_proj = nn.Linear(embeding, 4096)

        trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def forward(self, img_embed):
        bs = img_embed.shape[0]
        cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)
        local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local),
                                          key=self.norm_layer_aggr(img_embed[:, :, :]),
                                          value=self.norm_layer_aggr(img_embed[:, :, :]))[0]

        output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
        atts_local_feat_aggr = self.bbox_proj(local_feat_aggr)
        return output_coord,atts_local_feat_aggr





