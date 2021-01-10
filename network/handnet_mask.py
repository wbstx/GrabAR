import torch
import torch.nn.functional as F
from torch import nn

from global_context_box import ContextBlock2d

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class SplitBN(nn.Module):
    def __init__(self, inplanes):
        super(SplitBN, self).__init__()

        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight.data.fill_(1.)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias.data.fill_(0.0)

        num_groups = 32 if inplanes >= 32 else 16
        self.bn_syn = nn.GroupNorm(num_groups=num_groups, num_channels=inplanes, affine=False)
        self.bn_real = nn.GroupNorm(num_groups=num_groups, num_channels=inplanes, affine=False)
    
    def forward(self, datas, syn_index):

        batch_size = datas.shape[0]
        if syn_index != 0 and syn_index != batch_size:
            split = torch.split(datas, [syn_index, batch_size-syn_index], dim=0)
            syn_normed = self.bn_syn(split[0])
            real_normed = self.bn_real(split[1])
            datas = torch.cat((syn_normed, real_normed), dim=0)
            datas = self.weight * datas + self.bias
        # no syn datas
        elif syn_index == 0:
            real_normed = self.bn_real(datas)
            datas = self.weight * real_normed + self.bias
        # no real datas
        elif syn_index == batch_size:
            syn_normed = self.bn_syn(datas)
            datas = self.weight * syn_normed + self.bias

        return datas

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, padding=0, stride=1, dilation=1, bias=True):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
        self.split_bn = SplitBN(outplanes)
        num_groups = 32 if outplanes >= 32 else 16
        self.bn = nn.GroupNorm(num_groups=num_groups, num_channels=outplanes)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, datas, labels):
        out = self.conv(datas)
        # out = self.split_bn(out, labels)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out

class PredictBlock(nn.Module):
    def __init__(self, inplanes, has_dropout=False):
        super(PredictBlock, self).__init__()
        self.conv1 = ConvBlock(inplanes, 16, 3, padding=1, bias=False)
        self.has_dropout = has_dropout
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(16, 3, 1)
    
    def forward(self, datas):
        out = self.conv1(datas, 0)
        if self.has_dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        return F.log_softmax(out, dim=1)

class ConvTransposeBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, padding=0, stride=1):
        super(ConvTransposeBlock, self).__init__()
        
        self.conv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride=stride, padding=padding)
        self.split_bn = SplitBN(outplanes)
        num_groups = 32 if outplanes >= 32 else 16
        self.bn = nn.GroupNorm(num_groups=num_groups, num_channels=outplanes)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, datas, labels):
        out = self.conv(datas)
        # out = self.split_bn(out, labels)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, inplanes, outplanes, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(inplanes, outplanes, 3, stride=1, padding=1, bias=False)
        # self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, datas):
        out = nn.functional.interpolate(datas, scale_factor=self.scale_factor, mode='bilinear')
        out = self.conv(out)
        # out = self.leaky_relu(out)
        return out

class FusionBlock(nn.Module):
    def __init__(self, inplanes):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, 3, stride=1, padding=1)
        num_groups = 32 if inplanes >= 32 else 16
        self.bn = nn.GroupNorm(num_groups=num_groups, num_channels=inplanes)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, datas):
        out = self.conv(datas)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out

class HandNetInitial(nn.Module):
    def __init__(self):
        super(HandNetInitial, self).__init__()

        # self.aspp = build_aspp('unet', 8, nn.BatchNorm2d)
        self.down1 = ConvBlock(6, 16, 3, padding=1, stride=1)
        self.down2 = ConvBlock(16, 32, 3, padding=1, stride=2)
        self.down4 = ConvBlock(32, 64, 4, padding=1, stride=2)
        self.down8 = ConvBlock(64, 128, 4, padding=1, stride=2)
        self.down16 = ConvBlock(128, 256, 5, padding=2, stride=2)
        self.down32 = ConvBlock(256, 512, 5, padding=2, stride=2)

        self.dial1 = ConvBlock(512, 512, 3, padding=2, dilation=2)
        self.dial2 = ConvBlock(512, 512, 3, padding=4, dilation=4)
        self.dial3 = ConvBlock(512, 512, 3, padding=2, dilation=2)
        # self.psp = PSPModule(256, 256)
        self.global_context1 = ContextBlock2d(512, 128, pool='att', fusions=['channel_add'])
        self.global_context2 = ContextBlock2d(256, 64, pool='att', fusions=['channel_add'])
        self.global_context3 = ContextBlock2d(128, 32, pool='att', fusions=['channel_add'])

        self.upsample32 = UpsampleBlock(512, 256, scale_factor=2)
        self.upsample16 = UpsampleBlock(256, 128, scale_factor=2)
        self.upsample8 = UpsampleBlock(128, 64, scale_factor=2)
        self.upsample4 = UpsampleBlock(64, 32, scale_factor=2)
        self.upsample2 = UpsampleBlock(32, 16, scale_factor=2)

        # self.global_upsample2 = UpsampleBlock(128, 64, scale_factor=2)
        # self.global_upsample4 = UpsampleBlock(128, 32, scale_factor=4)
        # self.global_upsample8 = UpsampleBlock(128, 16, scale_factor=8)
        # self.global_upsample16 = UpsampleBlock(128, 16, scale_factor=8)

        self.predict16 = PredictBlock(256, has_dropout=True)
        self.predict8 = PredictBlock(128, has_dropout=True)
        self.predict4 = PredictBlock(64, has_dropout=True)
        self.predict2 = PredictBlock(32, has_dropout=True)
        self.predict1 = PredictBlock(16, has_dropout=True)

        self.fusion32 = FusionBlock(256)
        self.fusion16 = FusionBlock(128)
        self.fusion8 = FusionBlock(64)
        self.fusion4 = FusionBlock(32)
        self.fusion2 = FusionBlock(16)

        self.predict = ConvBlock(16, 16, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.tail = nn.Conv2d(16, 3, 1)

    ##############################
    ## The batch is reordered by syn - real
    ## returns how many syn datas in the batch
    def reorder_data(self, x, y, target, labels):
        syn_indexs = torch.nonzero(1-labels, as_tuple=True)
        real_indexs = torch.nonzero(labels, as_tuple=True)

        if len(syn_indexs[0]) != 0 and len(real_indexs[0]) != 0:
            x_syn_datas = torch.index_select(x, dim=0, index=syn_indexs[0])
            x_real_datas = torch.index_select(x, dim=0, index=real_indexs[0])
            x = torch.cat((x_syn_datas, x_real_datas), dim=0)

            y_syn_datas = torch.index_select(y, dim=0, index=syn_indexs[0])
            y_real_datas = torch.index_select(y, dim=0, index=real_indexs[0])
            y = torch.cat((y_syn_datas, y_real_datas), dim=0)

            target_syn_datas = torch.index_select(target, dim=0, index=syn_indexs[0])
            target_real_datas = torch.index_select(target, dim=0, index=real_indexs[0])
            target = torch.cat((target_syn_datas, target_real_datas), dim=0)

        return x, y, target, len(syn_indexs[0])

    def forward(self, x, y, masks, label):

        # x = self.split_bn1(x, label)
        # Encoding
        # x, y, target, syn_index = self.reorder_data(x, y, target, label)
        syn_index = 0

        # Hard coded mask
        # Overlapping Regions are 1 otherwise 0
        mask = torch.sign(torch.abs(x - y))

        f_d1 = self.down1(torch.cat((x, y), dim=1), syn_index)
        f_d2 = self.down2(f_d1, syn_index)
        f_d4 = self.down4(f_d2, syn_index)
        f_d8 = self.down8(f_d4, syn_index)
        f_d16 = self.down16(f_d8, syn_index)
        f_d32 = self.down32(f_d16, syn_index)

        f_a1 = self.dial1(f_d32, syn_index)
        f_a2 = self.dial2(f_a1, syn_index)
        f_a3 = self.dial3(f_a2, syn_index)
        f_a3, f_global = self.global_context1(f_a3)

        # global_feature = self.psp(d_f7)                         # Which has the biggest reception field
        f_u16 = self.fusion32(self.upsample32(f_a3) + f_d16)
        p16 = self.predict16(f_u16)
        result16_max, _ = torch.max(p16, 1)
        result16_max = 2 - torch.exp(result16_max) * 1.5
        result16_max = torch.unsqueeze(result16_max, 1)
        result16_max = nn.functional.interpolate(result16_max, scale_factor=2, mode='bilinear')
        f_u16, _ = self.global_context2(f_u16)

        mask_temp = nn.AdaptiveAvgPool2d((40, 40))(mask[:,2,:,:]) + 1
        mask_temp = torch.unsqueeze(mask_temp, 1)
        # f_u8 = self.fusion16(self.upsample16(f_u16) + f_d8)
        f_u8 = self.fusion16(self.upsample16(f_u16) + f_d8 * result16_max)
        # f_u8 = self.fusion16(self.upsample16(f_u16) + f_d8 * result16_max * mask_temp)
        p8 = self.predict8(f_u8)
        result8_max, _ = torch.max(p8, 1)
        result8_max = 2 - torch.exp(result8_max) * 1.5
        result8_max = torch.unsqueeze(result8_max, 1)
        result8_max = nn.functional.interpolate(result8_max, scale_factor=2, mode='bilinear')
        f_u8, _ = self.global_context3(f_u8)

        mask_temp = nn.AdaptiveAvgPool2d((80, 80))(mask[:,2,:,:]) + 1
        mask_temp = torch.unsqueeze(mask_temp, 1)
        # f_u4 = self.fusion8(self.upsample8(f_u8)  + f_d4 * result8_max * mask_temp)
        f_u4 = self.fusion8(self.upsample8(f_u8)  + f_d4 * result8_max)
        # f_u4 = self.fusion8(self.upsample8(f_u8) + f_d4)
        p4 = self.predict4(f_u4)
        result4_max, _ = torch.max(p4, 1)
        result4_max = 2 - torch.exp(result4_max) * 1.5
        result4_max = torch.unsqueeze(result4_max, 1)
        result4_max = nn.functional.interpolate(result4_max, scale_factor=2, mode='bilinear')

        mask_temp = nn.AdaptiveAvgPool2d((160, 160))(mask[:,2,:,:]) + 1
        mask_temp = torch.unsqueeze(mask_temp, 1)
        # f_u2 = self.fusion4(self.upsample4(f_u4)  + f_d2 * result4_max * mask_temp)
        f_u2 = self.fusion4(self.upsample4(f_u4)  + f_d2 * result4_max)
        # f_u2 = self.fusion4(self.upsample4(f_u4) + f_d2)
        p2 = self.predict2(f_u2)
        result2_max, _ = torch.max(p2, 1)
        result2_max = 2 - torch.exp(result2_max) * 1.5
        result2_max = torch.unsqueeze(result2_max, 1)
        result2_max = nn.functional.interpolate(result2_max, scale_factor=2, mode='bilinear')

        mask_temp = mask[:,2,:,:] + 1
        mask_temp = torch.unsqueeze(mask_temp, 1)
        # f_u1 = self.fusion2(self.upsample2(f_u2)  + f_d1 * result2_max * mask_temp)
        f_u1 = self.fusion2(self.upsample2(f_u2)  + f_d1 * result2_max)
        # f_u1 = self.fusion2(self.upsample2(f_u2) + f_d1)
        p1 = self.predict1(f_u1)

        # d_f8 = self.fusion8(self.upsample8(d_f7 + d_f4) + self.global_upsample2(global_feature))
        # d_f9 = self.fusion9(self.upsample9(d_f8 + d_f3) + self.global_upsample4(global_feature))
        # d_f10 = self.fusion10(self.upsample10(d_f9 + d_f2) + self.global_upsample8(global_feature))

        # d_f8 = self.conv8(d_f7, syn_index) + d_f3                # (1 * 128 * 128 * 128)
        # d_f9 = self.conv9(d_f8, syn_index) + d_f2                # (1 * 64 * 256 * 256)
        # d_f10 = self.conv10(d_f9, syn_index) + d_f1              # (1 * 32 * 512 * 512)

        # pre = self.predict(f_u1, syn_index)              # (1 * 32 * 512 * 512)
        # pre = self.dropout(pre)
        # res = self.tail(pre)                        # (1 * 2 * 512 * 512)

        return [p1, p2, p4, p8, p16], mask[:, 2, :, :]
        # return F.softmax(res, dim=1)


