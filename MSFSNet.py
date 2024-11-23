import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.arch_utils import LayerNorm2d, MySequential


class SFF(nn.Module):
    #skip feature fusion
    def __init__(self, in_channels, height=2,reduction=8, bias=False):
        super().__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias),SimpleGate())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d//2, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels*4,in_channels,1)
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels*2, in_channels * 4, 1, bias=False),
                    nn.PixelShuffle(2)
                )

    def forward(self, f_r, f_m):

        f_m = self.conv2(f_m)
        f_r = self.conv1(f_r)
        feats_U = f_r + f_m
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        a_r = self.softmax(self.fcs[0](feats_Z))
        a_m = self.softmax(self.fcs[1](feats_Z))
      
        
        m = f_m * a_m + f_m
        r = f_r * a_r + f_r + m
        
        return r, m



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
    
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(LayerNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
class SFE(nn.Module):
    def __init__(self, out_plane):
        super(SFE, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

      


    def forward(self, x):
        x = self.main(x)
        return x
    

class dfs(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dfs, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.ln = LayerNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.modulate = FCAM(inchannels)

    def forward(self, x):
        identity_input = x 
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.ln(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out

class FCAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_h):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_h_T = self.r_proj1(self.norm_r(x_h)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_h = self.r_proj2(x_h).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_h_T) * self.scale

        F_h2l = torch.matmul(torch.softmax(attention, dim=-1), V_h)  #B, H, W, c
        F_l2h = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_h2l = F_h2l.permute(0, 3, 1, 2) * self.beta
        F_l2h = F_l2h.permute(0, 3, 1, 2) * self.gamma
        return F_h2l + F_l2h



    

class MD(nn.Module):
    #multi-scal frequency select (MFSNet)
    def __init__(self, in_channel):
        super().__init__()

        self.project_in = nn.Conv2d(in_channel, in_channel, kernel_size=1)
       
        self.dyna = dfs(in_channel//2) 
        self.dyna_2 = dfs(in_channel//2, kernel_size=5) 
        self.project_out = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, x):
        out = self.project_in(x)
       
        k3, k5 = torch.chunk(out, 2, dim=1)
        out_k3 = self.dyna(k3)
        out_k5 = self.dyna_2(k5)
        out = torch.cat((out_k3, out_k5), dim=1)
        out = self.project_out(out)
        return out + x



class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)],
                    MD(chan)
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)],
                MD(chan)
            )
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)],
                    MD(chan)
                )
            )
           

        self.padder_size = 2 ** len(self.encoders)

        self.SFE2 = SFE(width*2)
        self.SFE4 = SFE(width*4)
        self.SFE8 = SFE(width*8)

        self.FAM2 = FAM(width * 2)
        self.FAM4 = FAM(width * 4)
        self.FAM8 = FAM(width * 8)

        self.mid = nn.ModuleList()
        for i in range(len(enc_blk_nums)):
            self.mid_e = nn.ModuleList()
            for j in range(len(enc_blk_nums)):
                w = width
                if i<j:
                    self.mid_e.append( 
                        nn.Sequential(
                            nn.Conv2d(w*2**j, w * 2**(2*j-i) , 1, bias=False),
                            nn.PixelShuffle(2**(j-i))
                    )
                )
                elif i>j:
                    self.mid_e.append( 
                       nn.Conv2d(w*2**j, w*2**i, 1, 2**(i-j))
                    )
            self.mid.append(self.mid_e)

        self.midout = nn.ModuleList()
        en_number = len (enc_blk_nums)
        for i in range(en_number):
            self.mid_e = nn.ModuleList()
            for j in range(en_number):
                w = width
                if i<j:
                    self.mid_e.append( 
                       nn.Conv2d(w*2**(en_number-j-1), w*2**(en_number-i-1), 1, 2**(j-i))
                    )
                elif i>j:
                    self.mid_e.append( 
                        nn.Sequential(
                            nn.Conv2d(w*2**(en_number-j-1), w * 2**(en_number-j-1+i-j) , 1, bias=False),
                            nn.PixelShuffle(2**(i-j))
                    )
                )  
            self.midout.append(self.mid_e)
        
        self.skff = nn.ModuleList()
        for i in range(len(enc_blk_nums)):
            self.skff.append(
                        SFF(width*2**i)
                )
                
        self.Convs = nn.ModuleList([
            nn.Conv2d(width * 16, width * 8, kernel_size=1),
            nn.Conv2d(width * 8, width*4, kernel_size=1,),
            nn.Conv2d(width * 4, width*2, kernel_size=1,),
            nn.Conv2d(width * 2, width, kernel_size=1,),
        ])

        self.Convs1 = nn.ModuleList([
            nn.Conv2d(width * 32, width * 8, kernel_size=1),
            nn.Conv2d(width * 16, width*4, kernel_size=1,),
            nn.Conv2d(width * 8, width*2, kernel_size=1,),
            nn.Conv2d(width * 4, width, kernel_size=1,),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                nn.Conv2d(width * 8, 3, kernel_size=3,padding=1, stride=1, groups=1,
                              bias=True),
                nn.Conv2d(width * 4, 3, kernel_size=3,padding=1, stride=1, groups=1,
                              bias=True),
                nn.Conv2d(width * 2, 3, kernel_size=3,padding=1, stride=1, groups=1,
                              bias=True),
            ]
        )

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x_2 = F.interpolate(inp, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        x_8 = F.interpolate(x_4, scale_factor=0.5)
        z2 = self.SFE2(x_2)
        z4 = self.SFE4(x_4)
        z8 = self.SFE8(x_8)

        x = self.intro(inp)

        encs = []
        enc_i = 0
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            enc_i = enc_i + 1
            if enc_i == 1:
                x = self.FAM2(x, z2)
            elif enc_i == 2:
                x = self.FAM4(x, z4)
            elif enc_i == 3:
                x = self.FAM8(x, z8)


        sig = []
        for i in range(len(self.encoders)):
            sig_en = []
            for j in range(len(self.encoders)):
                if i==j:
                    sig_en.append(encs[i])
                elif i<j:
                    sig_en.append(self.mid[i][j-1](encs[j]))
                else:
                    sig_en.append(self.mid[i][j](encs[j]))

            signal = torch.concat([s for s in sig_en],dim=1)
            sig.append(signal)
         
        x = self.middle_blks(x)
        global_feature = x
        skip_f = []

        for i in range(len(sig)):
            skip, global_feature = self.skff[len(sig)-i-1](sig[len(sig)-i-1], global_feature)
            skip_f.append(skip)
        
        skip_features = []
        for i in range(len(skip_f)):
            sig_en = []
            for j in range(len(skip_f)):
                if i==j:
                    sig_en.append(skip_f[i])

                elif i<j:

                    sig_en.append(self.midout[i][j-1](skip_f[j]))
                else:
                    sig_en.append(self.midout[i][j](skip_f[j]))

            signal = torch.concat([s for s in sig_en],dim=1)
            skip_features.append(signal)

        

        index = 0
        decs = []
        for decoder, up, enc_skip in zip(self.decoders, self.ups, skip_features):
            x = up(x)
            enc_skip = self.Convs1[index](enc_skip)
            x = torch.concatenate([x,enc_skip],dim=1)
            x = self.Convs[index](x)
            index = index+1
            x = decoder(x)
            decs.append(x)
        
        outs = []
        outs.append(self.ConvsOut[0](decs[0])+x_8)
        outs.append(self.ConvsOut[1](decs[1])+x_4)
        outs.append(self.ConvsOut[2](decs[2])+x_2)

        x = self.ending(x)
        x = x + inp
        outs.append(x)

        return outs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)

            if m.output_size == 1:
                setattr(model, n, pool)
            # assert m.output_size == 1
            

        # if isinstance(m, Attention):
        #     attn = LocalAttention(dim=m.dim, num_heads=m.num_heads, is_prompt=m.is_prompt, bias=True, base_size=base_size, fast_imp=False,
        #                           train_size=train_size)
        #     setattr(model, n, attn)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)




if __name__ == "__main__":
    net = NAFNetLocal()
    x1 =  torch.randn((1, 3, 30, 90))
    x2 =  torch.randn((1, 3, 30, 90))
    x = torch.randn((2, 3, 256, 256))
    print("Total number of param  is ", sum(i.numel() for i in net.parameters()))
    t=net(x)
    print(len(t))
    inp_shape = (3, 256, 256)
    from ptflops import get_model_complexity_info
    FLOPS = 0

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    # print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9



    print('mac', macs, params)

    from thop import profile
    x3 =  torch.randn((1, 3, 256, 256))
    flops, params = profile(net, inputs=(x3, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

   
