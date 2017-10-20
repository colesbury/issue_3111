import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import *
from PIL import Image


class Model(object):
  def __init__(self):
     self.network = models.resnet18(pretrained=True).cuda()
     self.network.eval()
     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
     self.transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
     ])

  def detect(self, img):
     img = Image.fromarray(img)
     t = self.transform(img)
     v = Variable(t.unsqueeze(0).cuda(), volatile=True)
     res = self.network(v).squeeze(0)
     score, best = res.max(dim=0)
     return best.data[0]
