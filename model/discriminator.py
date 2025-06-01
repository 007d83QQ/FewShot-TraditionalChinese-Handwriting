import torch
from torch import nn

from model.block import DownsampleBlock
from collections import OrderedDict


class EnhancedDiscriminator(nn.Module):
    def __init__(self, writer_count: int, character_count: int, image_size: int):
        super(EnhancedDiscriminator, self).__init__()

        self.feature_extractor = nn.Sequential(
            DownsampleBlock(1, 64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            DownsampleBlock(256, 512),
            DownsampleBlock(512, 512),
            nn.Flatten(),
        )

        feature_size = 512 * (image_size // 32) ** 2
        self.reality_classifier = nn.Linear(feature_size, 1)
        self.writer_classifier = nn.Linear(feature_size, writer_count)
        self.character_classifier = nn.Linear(feature_size, character_count)

    def forward(self, input):
        feature = self.feature_extractor(input)

        reality = torch.sigmoid(self.reality_classifier(feature))
        writer = self.writer_classifier(feature)
        character = self.character_classifier(feature)

        return reality, writer, character


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, writer_count: int, character_count: int, num_scales: int = 3, image_size: int = 128):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_scales = num_scales

        network_list = []
        for _ in range(num_scales):
            network_list.append(EnhancedDiscriminator(writer_count, character_count, image_size))
            image_size //= 2
        self.network = nn.ModuleList(network_list)

        self.downsample = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

    def forward(self, input):
        output = []
        for i in range(self.num_scales):
            output.append(self.network[i](input))
            input = self.downsample(input)
        return output
    



class StyleTemplateDiscriminator(nn.Module):
    def __init__(self, img_channels=3, style_dim=512, n_heads=4):
        super().__init__()
        # CNN backbone
        self.feat = nn.Sequential(OrderedDict([
            ('block1', DownsampleBlock(img_channels, 64)),
            ('drop1',  nn.Dropout2d(0.1)),
            ('block2', DownsampleBlock(64, 128)),
            ('drop2',  nn.Dropout2d(0.1)),
            ('block3', DownsampleBlock(128, 256)),
            ('relu',   nn.ReLU(inplace=False)),
        ]))                                 # out→ (B,256,h,w)
        self.conv_q = nn.utils.spectral_norm(
            nn.Linear(style_dim, 256))

        self.attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=n_heads, batch_first=True, dropout=0.1)

        self.out = nn.Conv2d(256, 1, 3, 1, 1)  # PatchGAN logits
        
    def forward(self, gen_img, tmpl_img, style_vec):
        diff = torch.abs(gen_img - tmpl_img)
        x = torch.cat([gen_img, tmpl_img, diff], dim=1)   # B,3,H,W
        feat = self.feat(x)                               # B,C,h,w
        B,C,h,w = feat.shape
        q = feat.flatten(2).permute(0,2,1)            # B,h*w,C
        k =v= self.conv_q(style_vec).unsqueeze(1)           # B,1,C
        fused,_ = self.attn(q, k, v)                      # B,1,C
        fused = fused.expand(-1, h*w, -1).transpose(1,2).reshape(B,C,h,w)
        logits = self.out(fused)                          # B,1,h,w
        return logits