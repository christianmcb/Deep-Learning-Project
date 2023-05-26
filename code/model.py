import torchvision
import torch

class Conv2dBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
        self.bn1 = torch.nn.BatchNorm2d(out_c)
        self.conv2 = torch.nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')
        self.bn2 = torch.nn.BatchNorm2d(out_c)
        self.relu = torch.nn.ReLU()
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Conv2dTransposeBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=(3,3))
        self.bn1 = torch.nn.BatchNorm2d(out_c)
        self.conv2 = torch.nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_c)
        self.relu = torch.nn.ReLU()
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

#conv = torch.nn.Conv2d(in_channels= 3, out_channels= 16, kernel_size=(3,3), padding='same')

class UNet(torch.nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        channels = [3, 16, 32, 64, 128, 256]

        # Conolution layers where conv2d_1 = First Layer
        self.conv2d_1 = Conv2dBlock(in_c=channels[0],out_c=channels[1])
        self.conv2d_2 = Conv2dBlock(in_c=channels[1],out_c=channels[2])
        self.conv2d_3 = Conv2dBlock(in_c=channels[2],out_c=channels[3])
        self.conv2d_4 = Conv2dBlock(in_c=channels[3],out_c=channels[4])
        self.conv2d_5 = Conv2dBlock(in_c=channels[4],out_c=channels[5])

        # Decoding layers for upsampling where conv2dTranspose_1 = First Layer
        self.conv2dTranspose_1 = torch.nn.ConvTranspose2d(in_channels=channels[5], out_channels=channels[4], kernel_size=3, stride=2, padding=1)
        self.conv2d_6 = Conv2dBlock(in_c=channels[5], out_c=channels[4])
        self.conv2dTranspose_2 = torch.nn.ConvTranspose2d(in_channels=channels[4], out_channels=channels[3], kernel_size=3, stride=2, padding=1)
        self.conv2d_7 = Conv2dBlock(in_c=channels[4], out_c=channels[3])
        self.conv2dTranspose_3 = torch.nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=3, stride=2, padding=1)
        self.conv2d_8 = Conv2dBlock(in_c=channels[3], out_c=channels[2])
        self.conv2dTranspose_4 = torch.nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1], kernel_size=3, stride=2, padding=1)
        self.conv2d_9 = Conv2dBlock(in_c=channels[2], out_c=channels[1])

        # Define max pooling and dropout functions
        self.maxPool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        
        # Classify prediction mask to single channel
        self.segment = torch.nn.Conv2d(channels[1], 1, kernel_size=1, padding=0)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        # Encoding block
        enc = []
        for conv in [self.conv2d_1, self.conv2d_2, self.conv2d_3, self.conv2d_4]:
            x = conv(x)
            enc.append(x)
            x = self.maxPool(x)
            #x = self.dropout(x)
        x = self.conv2d_5(x)

        # Decoding block
        for i, l in enumerate([[self.conv2dTranspose_1, self.conv2d_6], [self.conv2dTranspose_2, self.conv2d_7], 
                                [self.conv2dTranspose_3,self.conv2d_8], [self.conv2dTranspose_4, self.conv2d_9]]):
            trans, conv = l[0], l[1]
            x = trans(x, output_size=((x.size()[2])*(2), x.size()[3]*(2)))
            x = torch.cat((x, enc[3-i]), axis=1)
            #x = self.dropout(x)
            x = conv(x)
        
        x = self.segment(x)
        x = torch.squeeze(x)
        #x = self.activation(x)

        return x

def DeepLabModel():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 1 class.
    return model

def DeepLabV3_MobileNet_V3_Large():
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, num_classes=1)
    return model
