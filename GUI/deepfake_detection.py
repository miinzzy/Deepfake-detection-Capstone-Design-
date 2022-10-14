import easydict
import os
import sys
from PIL import Image
import tqdm
import shutil
import cv2

import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms

##
seed = 719
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
##

##
cudnn.benchmark = True

def deepfake_detection(valid_path):

    args = easydict.EasyDict({
        "gpu": 0,
        "valid_path": valid_path,
        "save_fn": "C:\\deepfake\\model\\xception_new_model.pth.tar",
    })

    assert os.path.isfile(args.valid_path), 'wrong path'
    ##

    ##
    class SeparableConv2d(nn.Module):
        def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
            super(SeparableConv2d,self).__init__()

            self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
            self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

        def forward(self,x):
            x = self.conv1(x)
            x = self.pointwise(x)
            return x


    class Block(nn.Module):
        def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
            super(Block, self).__init__()

            if out_filters != in_filters or strides!=1:
                self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
                self.skipbn = nn.BatchNorm2d(out_filters)
            else:
                self.skip=None

            self.relu = nn.ReLU(inplace=True)
            rep=[]

            filters=in_filters
            if grow_first:
                rep.append(self.relu)
                rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
                rep.append(nn.BatchNorm2d(out_filters))
                filters = out_filters

            for i in range(reps-1):
                rep.append(self.relu)
                rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
                rep.append(nn.BatchNorm2d(filters))

            if not grow_first:
                rep.append(self.relu)
                rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
                rep.append(nn.BatchNorm2d(out_filters))

            if not start_with_relu:
                rep = rep[1:]
            else:
                rep[0] = nn.ReLU(inplace=False)

            if strides != 1:
                rep.append(nn.MaxPool2d(3,strides,1))
            self.rep = nn.Sequential(*rep)

        def forward(self,inp):
            x = self.rep(inp)

            if self.skip is not None:
                skip = self.skip(inp)
                skip = self.skipbn(skip)
            else:
                skip = inp

            x+=skip
            return x


    class Xception(nn.Module):
        def __init__(self, num_classes=1000):
            super(Xception, self).__init__()
            self.num_classes = num_classes

            self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(32,64,3,bias=False)
            self.bn2 = nn.BatchNorm2d(64)

            self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
            self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
            self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

            self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

            self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

            self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

            self.conv3 = SeparableConv2d(1024,1536,3,1,1)
            self.bn3 = nn.BatchNorm2d(1536)

            self.conv4 = SeparableConv2d(1536,2048,3,1,1)
            self.bn4 = nn.BatchNorm2d(2048)

            self.fc = nn.Linear(2048, num_classes)

        def features(self, input):
            x = self.conv1(input)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = self.conv4(x)
            x = self.bn4(x)
            return x

        def logits(self, features):
            x = self.relu(features)

            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.last_linear(x)
            return x

        def forward(self, input):
            x = self.features(input)
            x = self.logits(x)
            return x


    ## 기존 Xception에 Dropout만 추가
    class xception(nn.Module):
        def __init__(self, num_out_classes=2, dropout=0.5):
            super(xception, self).__init__()

            self.model = Xception(num_classes=num_out_classes)
            self.model.last_linear = self.model.fc
            del self.model.fc

            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )

        def forward(self, x):
            x = self.model(x)
            return x
    ##

    ##
    xception_default = {
        'train': transforms.Compose([transforms.CenterCrop((299, 299)),
                                     transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalize([0.5]*3, [0.5]*3),
                                     ]),
        'valid': transforms.Compose([transforms.CenterCrop((299, 299)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5]*3, [0.5]*3),
                                     ]),
        'test': transforms.Compose([transforms.CenterCrop((299, 299)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5] * 3, [0.5] * 3),
                                    ]),
    }
    ##

    # custom dataset

    class ImageRecord(object):
        def __init__(self, row):
            self._data = row

        @property
        def path(self):
            return self._data[0]

        @property
        def label(self):
            return int(self._data[1])


    class DFDCDatatset(data.Dataset):
        def __init__(self, valid_path, transform=None):
            # self.root_path = root_path
            self.valid_path = valid_path
            self.transform = transform

            # self._parse_list()

        def _load_image(self, image_path):
            return Image.open(image_path).convert('RGB')

        # def _parse_list(self):
        #     self.image_list = [ImageRecord(x.strip().split(' ')) for x in open(self.list_file)]

        def __getitem__(self, index):
            # record = self.image_list[index]
            image_name = os.path.join(self.valid_path)
            unnorm_image = self._load_image(image_name)
            unnorm_crop_image = transforms.Compose([transforms.CenterCrop(size=(299, 299)),
                                                    transforms.ToTensor()])(unnorm_image)

            if self.transform is not None:
                image = self.transform(unnorm_image)

            return image, unnorm_crop_image

        def __len__(self):
            return 1
    ##

    finalconv_name = 'block12'
    feature_blobs = []
    backward_feature = []

    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data)

    def backward_hook(module, input, output):
        backward_feature.append(output[0])

    ##
    # validate

    def validate(test_loader, model, criterion, count):

        classnum = 2
        model.eval()

        cnt = 0

        with tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Valid", file=sys.stdout) as iterator:
            for images, unnorm_crop_image in iterator:
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    # target = target.cuda(args.gpu, non_blocking=True)

                output = model(images)
                _, pred = torch.max(output.data, 1)

                # --------------------------------<Grad-CAM>--------------------------------------#
                model.model._modules.get(finalconv_name).register_forward_hook(hook_feature)
                model.model._modules.get(finalconv_name).register_backward_hook(backward_hook)

                score = output.squeeze()[1]  # [1, 2] -> [1] (probability of FAKE)
                score.backward(retain_graph=True)

                if len(feature_blobs) > 0 & len(backward_feature):
                    activations = feature_blobs[0].to(args.gpu)  # [1, 64, 147, 147]
                    gradients = backward_feature[0]  # [1, 64, 147, 147]
                    b, k, u, v = gradients.size()

                    alpha = gradients.view(b, k, -1).mean(2)  # [1, 64, 147, 147] -> [1, 64]
                    # print('\nalpha shape: ', alpha.shape)
                    weights = alpha.view(b, k, 1, 1)  # [1, 64, 1, 1]
                    # print('weights shape: ', weights.shape)

                    grad_cam_map = (weights * activations).sum(1,
                                                               keepdim=True)  # alpha * A^k = (1, 64, 147, 147) => (1, 1, 147, 147)
                    grad_cam_map = F.relu(grad_cam_map)  # Apply R e L U
                    grad_cam_map = F.interpolate(grad_cam_map, size=(299, 299), mode='bilinear',
                                                 align_corners=False)  # (1, 1, 299, 299)
                    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
                    grad_cam_map = (grad_cam_map - map_min).div(
                        map_max - map_min).data  # (1, 1, 299, 299), min-max scaling

                    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()),
                                                     cv2.COLORMAP_JET)  # (299, 299, 3)
                    grad_heatmap = np.float32(grad_heatmap) / 255
                    grad_heatmap = cv2.cvtColor(grad_heatmap, cv2.COLOR_RGB2BGR)

                    img = np.transpose(unnorm_crop_image.cpu().numpy().squeeze(),
                                       (1, 2, 0))  # (1, 3, 299, 299) -> (299, 299, 3)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    global grad_result

                    grad_result = 0.7 * grad_heatmap + img
                    grad_result = grad_result / np.max(grad_result)
                    grad_result = np.uint8(255 * grad_result)

        if count == 1:
            return output
        elif count == 2:
            return grad_result

    ##

    ##
    model = xception(num_out_classes=2, dropout=0.5)
    print("=> creating model '{}'".format('xception'))
    model = model.cuda(args.gpu)

    assert os.path.isfile(args.save_fn), 'wrong path'

    model.load_state_dict(torch.load(args.save_fn)['state_dict'])
    print("=> model weight '{}' is loaded".format(args.save_fn))

    model = model.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    ##

    ##
    valid_dataset = DFDCDatatset(
                                 args.valid_path,
                                 xception_default["test"],
                                 )
    ##

    ##
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               shuffle=False,
                                               pin_memory=False,
                                               )
    ##

    ##
    count = 1
    output = validate(valid_loader, model, criterion, count)
    count += 1
    grad_cam = validate(valid_loader, model, criterion, count)
    # result_grad_cam = cv2.cvtColor(grad_cam, cv2.COLOR_BGR2RGB)
    ##
    def center_crop(img, set_size):
        h, w, c = img.shape

        if set_size > min(h, w):
            return img

        crop_width = set_size
        crop_height = set_size

        mid_x, mid_y = w // 2, h // 2
        offset_x, offset_y = crop_width // 2, crop_height // 2

        crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
        return crop_img
    ##
    img = cv2.imread(args.valid_path)
    cropimg = center_crop(img, 299)
    cropimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2RGB)
    ##
    y = F.softmax(output[0], dim=0)
    ##

    if output[0][0] > output[0][1]:
        result = "REAL Image \n(Fake일 확률 : " + str(round(y[1].item() * 100, 3)) + "%)"
    else:
        result = "FAKE Image \n(Fake일 확률 : " + str(round(y[1].item() * 100, 3)) + "%)"

    return result, cropimg, grad_cam
