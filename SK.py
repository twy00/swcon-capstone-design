import torch
import torch.nn as nn
import torch.nn.functional as F
#from FCSN import FCSN
import numpy as np

import random
class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p

class SK(nn.Module):
    def __init__(self, n_class=2):
        super(SK, self).__init__()
        self.conv1_1 = nn.Conv2d(1024, 64, (1,3), padding=(0,100))
        self.sn1_1 = nn.utils.spectral_norm(self.conv1_1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, (1,3), padding=(0,1))
        self.sn1_2 = nn.utils.spectral_norm(self.conv1_2)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, (1,3), padding=(0,1))
        self.sn2_1 = nn.utils.spectral_norm(self.conv2_1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, (1,3), padding=(0,1))
        self.sn2_2 = nn.utils.spectral_norm(self.conv2_2)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, (1,3), padding=(0,1))
        self.sn3_1 = nn.utils.spectral_norm(self.conv3_1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, (1,3), padding=(0,1))
        self.sn3_2 = nn.utils.spectral_norm(self.conv3_2)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, (1,3), padding=(0,1))
        self.sn3_3 = nn.utils.spectral_norm(self.conv3_3)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, (1,3), padding=(0,1))
        self.sn4_1 = nn.utils.spectral_norm(self.conv4_1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn4_2 = nn.utils.spectral_norm(self.conv4_2)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn4_3 = nn.utils.spectral_norm(self.conv4_3)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn5_1 = nn.utils.spectral_norm(self.conv5_1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn5_2 = nn.utils.spectral_norm(self.conv5_2)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn5_3 = nn.utils.spectral_norm(self.conv5_3)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, (1,7))
        self.sn6 = nn.utils.spectral_norm(self.fc6)
        self.in6 = nn.InstanceNorm2d(4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=0.5)

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, (1,1))
        self.sn7 = nn.utils.spectral_norm(self.fc7)
        self.in7 = nn.InstanceNorm2d(4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.5)

        self.score_fr = nn.Conv2d(4096, n_class, (1,1))
        self.sn_score_fr = nn.utils.spectral_norm(self.score_fr)
        self.bn_score_fr = nn.BatchNorm2d(n_class)
        self.in_score_fr = nn.InstanceNorm2d(n_class)
        self.relu_score_fr = nn.ReLU(inplace=True)
        self.score_pool4 = nn.Conv2d(512, n_class, (1,1))
        self.sn_score_pool4 = nn.utils.spectral_norm(self.score_pool4)
        self.bn_score_pool4 = nn.BatchNorm2d(n_class)
        self.relu_bn_score_pool4 = nn.ReLU(inplace=True)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, (1,4), stride=(1,2))
        self.sn_upscore2 = nn.utils.spectral_norm(self.upscore2)
        self.bn_upscore2 = nn.BatchNorm2d(n_class)
        self.relu_upscore2 = nn.ReLU(inplace=True)

        self.upscore16 = nn.ConvTranspose2d(
            n_class, n_class, (1,32), stride=(1,16))
        self.sn_upscore16 = nn.utils.spectral_norm(self.upscore16)
        self.bn_upscore16 = nn.BatchNorm2d(n_class)
        self.relu_upscore16 = nn.ReLU(inplace=True)
        self.sigmoid_upscore16 = nn.Sigmoid()
        self.tanh_upscore16 = nn.Tanh()

        self.relu_add = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

        self.conv_reconstuct1 = nn.Conv2d(n_class, 1024, (1,1))
        self.bn_reconstruct1 = nn.BatchNorm2d(1024)
        self.relu_reconstuct1 = nn.ReLU(inplace=True)

        self.conv_reconstuct2 = nn.Conv2d(1024, 1024, (1,1))
        self.bn_reconstruct2 = nn.BatchNorm2d(1024)
        self.relu_reconstuct2 = nn.ReLU(inplace=True)


    def forward(self, x):
        # input
        h = x
        in_x = x
        # conv1
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))       
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))       
        h = self.pool1(h)                                   
        # conv2
        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))       
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))       
        h = self.pool2(h)                                   
        # conv3
        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))       
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))       
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))       
        h = self.pool3(h)                                   
        # conv4
        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))       
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))       
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))       
        h = self.pool4(h)                                   
        pool4 = h
        # conv5
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))       
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))       
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))       
        h = self.pool5(h)                                   
        # conv6
        h = self.relu6(self.fc6(h))                         
        h = self.drop6(h)                                   
        # conv7
        h = self.relu7(self.fc7(h))                         
        h = self.drop7(h)                                   
        # conv8
        h = self.in_score_fr(self.score_fr(h)) 
        
        
        # deconv1
        h = self.upscore2(h)
        upscore2 = h
        
        h = self.bn_score_pool4(self.score_pool4(pool4))
        h = h[:, :, :, 5:5+upscore2.size()[3]]
        score_pool4c = h
        
        h = upscore2+score_pool4c
        # deconv2
        h = self.upscore16(h)
        h = h[:, :, :, 27:27+x.size()[3]]; 

        # h
        h_softmax = self.softmax(h); 

        
        mask = h_softmax[:,1,:].view(1,1,1,-1); 

        h_mask = h*mask; 

        h_reconstruct = self.relu_reconstuct1(self.bn_reconstruct1(self.conv_reconstuct1(h_mask))) 
        x_select = in_x*mask

        h_merge = h_reconstruct + x_select 
        h_merge_reconstruct = self.relu_reconstuct2(self.bn_reconstruct2(self.conv_reconstuct2(h_merge))) 

        t = torch.tensor(1).cuda()
        q = torch.tensor(0).cuda()

        mask_list = mask[0][0][0]
        mask_list_np = np.reshape(mask_list.cpu().data.numpy(), (-1))
        q85 = torch.tensor(np.percentile(mask_list_np, 85)).cuda()
        
        mean = torch.mean(mask_list)
        r = torch.where(mask_list > q85 , t, q)


        return h_merge_reconstruct,mask,h,r   ,[1,1,1,T],[1,2,1,T]

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(1),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class SK_test(nn.Module):
    def __init__(self, n_channels=1024, n_classes=2):
        super(SK_test, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.softmax = nn.Softmax(dim=1)


        self.conv_reconstuct1 = nn.Conv2d(n_classes, 1024, (1,1))
        self.bn_reconstruct1 = nn.BatchNorm2d(1024)
        self.relu_reconstuct1 = nn.ReLU(inplace=True)

        self.conv_reconstuct2 = nn.Conv2d(1024, 1024, (1,1))
        self.bn_reconstruct2 = nn.BatchNorm2d(1024)
        self.relu_reconstuct2 = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x

        x1 = self.inc(x);       
        x2 = self.down1(x1);    
        x3 = self.down2(x2);    
        x4 = self.down3(x3);    
        x5 = self.down4(x4);    
        x = self.up1(x5, x4);   
        x = self.up2(x, x3);    
        x = self.up3(x, x2);    
        x = self.up4(x, x1);    
        x = self.outc(x);       

        h_softmax = self.softmax(x)

        mask = h_softmax[:,1,:].view(1,1,1,-1); 

        h_mask = x*mask; 

        h_reconstruct = self.relu_reconstuct1(self.bn_reconstruct1(self.conv_reconstuct1(h_mask))) 
        x_select = h*mask

        h_merge = h_reconstruct + x_select 
        h_merge_reconstruct = self.relu_reconstuct2(self.bn_reconstruct2(self.conv_reconstuct2(h_merge))) 


        return h_merge_reconstruct,mask,x   ,[1,1,1,T],[1,2,1,T]
         


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = SK_test(1024, 2)
    model.to(device)
    inp = torch.randn(1, 1024, 1, 100, requires_grad=True).to(device)

    a,b,c = model(inp)
    print(a.shape)
    print(b.shape)
    print(c.shape)