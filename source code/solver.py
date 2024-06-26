import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net_cl
import csv
from torchvision import transforms
from torchvision.transforms import ToTensor
import time
from tensorboardX import SummaryWriter
import cv2
import time
from copy import deepcopy
from torchvision.models import resnet50
from tqdm import *

torch.backends.cudnn.deterministic = True
import torch
import torch.nn as nn


def similar_matrix1(q, k, temperature=0.1): 
    qfh, qfl, qfall, qbh, qbl, qball = torch.chunk(q, 6, dim=0)
    kfh, kfl, kfall, kbh, kbl, kball = torch.chunk(k, 6, dim=0)

    l_pos = torch.einsum('nc,kc->nk', [qfall, kball])
    l_neg = torch.einsum('nc,kc->nk', [kfall, qball])
    # print(l_pos.shape, l_neg.shape)
    return l_pos.mean() + l_neg.mean()

def similar_matrix2(q, k, temperature=0.1):  
    qfh, qfl, qfall, qbh, qbl, qball = torch.chunk(q, 6, dim=0)
    kfh, kfl, kfall, kbh, kbl, kball = torch.chunk(k, 6, dim=0)
    # negative logits: NxK
    l_pos = torch.einsum('nc,kc->nk', [qfl, kfh])
    l_neg = torch.einsum('nc,kc->nk', [qbl, kbh])

    return 2 - l_pos.mean() - l_neg.mean()


class MlpNorm(nn.Module):
    def __init__(self, dim_inp=256, dim_out=64):
        super(MlpNorm, self).__init__()
        dim_mid = min(dim_inp, dim_out)  # max(dim_inp, dim_out)//2
        # hidden layers
        linear_hidden = []  # nn.Identity()
    
        linear_hidden.append(nn.Linear(dim_inp, dim_mid))
        linear_hidden.append(nn.Dropout(p=0.2))
        linear_hidden.append(nn.BatchNorm1d(dim_mid))
        linear_hidden.append(nn.LeakyReLU())
        self.linear_hidden = nn.Sequential(*linear_hidden)
        self.linear_out = nn.Linear(dim_mid, dim_out)  

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return F.normalize(x, p=2, dim=-1)


class SIAM(nn.Module):

    def __init__(self,
                 encoder,
                 clloss1=similar_matrix2,
                 clloss2=similar_matrix1,
                 temperature=0.1,
                 proj_num_layers=2,
                 pred_num_layers=2,
                 proj_num_length=64,
                 **kwargs):
        super().__init__()
        self.loss1 = clloss1
        self.loss2 = clloss2
        self.encoder = encoder
        # self.__name__ = self.encoder.__name__

        self.temperature = temperature
        self.proj_num_layers = proj_num_layers
        self.pred_num_layers = pred_num_layers

        self.projector = MlpNorm(32, 128)
        self.predictor = MlpNorm(128, 128)

    def forward(self, img, **args):  
        out = self.encoder(img, **args)
        self.pred = self.encoder.pred
        self.feat = self.encoder.feat
        return self.feat.clone(), self.pred.detach(),out
    
    def regular(self, sampler, lab, feat1, pred1,fov=None):  
        feat = sampler.select(feat1, pred1, lab, fov)

        proj = self.projector(feat)
        self.proj = proj
        pred = self.predictor(proj)

        losSG1 = self.loss1(pred, proj.detach(), temperature=self.temperature)
        losSG2 = self.loss1(proj, pred.detach(), temperature=self.temperature)


        losSG3 = self.loss2(proj, pred.detach(), temperature=self.temperature)
        losSG4 = self.loss2(proj, pred.detach(), temperature=self.temperature)
 
        return (losSG2+losSG1)+(losSG3 + losSG4)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets, shapes):
        device = inputs.device

        pred_sigmoid = inputs.sigmoid()
        target = targets.type_as(inputs)
      
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
       ) * pt.pow(self.gamma) * (1+shapes.to(device))
        focal_weight = (self.weight * target + (1 - self.weight) * (1 - target)) * pt.pow(self.gamma) * (
                    1 + shapes.to(device))
        loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='none') * focal_weight
 
        return loss.mean()


def points_selection_half(feat, prob, true, card=512, **args):  # point selection by ranking
    assert len(feat.shape) == 2, 'feat should contains N*L two dims!'
    L = feat.shape[-1]
   
    feat = feat[true.view(-1, 1).repeat(1, L) > .5].view(-1, L)

   
    prob = prob[true > .5].view(-1)
    idx = torch.sort(prob, dim=-1, descending=False)[1]
    idx_l, idx_h = torch.chunk(idx, chunks=2, dim=0)

    sample1 = idx_h[torch.randperm(idx_h.shape[0])[:card]]
    sample2 = idx_l[torch.randperm(idx_l.shape[0])[:card]]
    sample3 = idx[torch.randperm(idx.shape[0])[:card]]
    
    h = torch.index_select(feat, dim=0, index=sample1)
   
    l = torch.index_select(feat, dim=0, index=sample2)
    all = torch.index_select(feat, dim=0, index=sample3)
   
    return h, l, all


def points_selection_hard(feat, prob, true, card=512, dis=100, **args):  # point selection by ranking
    assert len(feat.shape) == 2, 'feat should contains N*L two dims!'
    L = feat.shape[-1]
   
    feat = feat[true.view(-1, 1).repeat(1, L) > .5].view(-1, L)
  
    prob = prob[true > .5].view(-1)
    idx = torch.sort(prob, dim=-1, descending=True)[1]
    
    h = torch.index_select(feat, dim=0, index=idx[:card])
    l = torch.index_select(feat, dim=0, index=idx[-card:])
   
    return h, l


class SeqNet(nn.Module):  # Supervised contrastive learning segmentation network

    def __init__(self, type_net, type_seg, num_emb=128):
        super(SeqNet, self).__init__()

        self.fcn = type_net  
        self.seg = type_seg  

        self.projector = MlpNorm(32, num_emb)
        self.predictor = MlpNorm(num_emb, num_emb)

    def forward(self, x):
        aux = self.fcn(x)
        self.feat = self.fcn.feat 
        out = self.seg(self.feat) 
        self.pred = out

        if self.training: 
            if isinstance(aux, (tuple, list)):
                return [self.pred, aux[0], aux[1]]
            else:
                return [self.pred, aux]
        return self.pred 


class MLPSampler:
    func = points_selection_half

    def __init__(self, mode='hard', top=4, low=1, dis=0, num=512, select3=False, roma=False):
        self.top = top
        self.low = low
        self.dis = dis
        self.num = num
        self.roma = roma
        self.select = self.select3 if select3 else self.select2
        self.func = eval('points_selection_' + mode)

    @staticmethod
    def rand(*args):
           return MLPSampler(mode='rand', num=512).select(*args)

    @staticmethod
    def half(*args):
        return MLPSampler(mode='half', num=512).select(*args)

    def norm(self, *args, roma=False):
            
        args = [F.normalize(arg, dim=-1) for arg in args]
        if len(args) == 1:
            return args[0]
        return args

    def select3(self, feat, pred, true, mask=None, ksize=5):
           
        assert feat.shape[-2:] == true.shape[-2:], 'shape of feat & true donot match!'
        assert feat.shape[-2:] == pred.shape[-2:], 'shape of feat & pred donot match!'
           
        feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])

        true = true.round()
        back = (F.max_pool2d(true, (ksize, ksize), 1, ksize // 2) - true).round()

        fh, fl, fall = self.func(feat, pred, true, top=self.top, low=self.low, dis=self.dis, card=self.num)
        bh, bl, ball = self.func(feat, 1 - pred, back, top=self.top, low=self.low, dis=self.dis, card=self.num)
        return torch.cat([fh, fl, fall, bh, bl, ball], dim=0)



class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, gpus):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        # self.criterion = torch.nn.BCELoss()
        self.criterion = FocalLoss(gamma=2, weight=0.25)
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.prediction_path = config.prediction_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.gpus = gpus
        self.build_model()
        self.sampler = MLPSampler(mode='half', select3=True, roma=False, top=4, low=2, dis=4, num=512)

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == "ECSNet":
            model1 = U_Net_cl(img_ch=3, output_ch=1)
            model2 = U_Net_cl(img_ch=32, output_ch=1)
            model = SeqNet(model1, model2, num_emb=128)
            self.unet = SIAM(encoder=model, clloss1=similar_matrix2, proj_num_length=128)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)
    
        self.unet = torch.nn.DataParallel(self.unet.cuda(), device_ids=self.gpus, output_device=self.gpus[0])
        self.unet = self.unet.cuda()

    

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def load_pretrain_model(self):
        state_path = './U-RISC-Data-Code/cem500k_mocov2_resnet50_200ep.pth.tar'
        state = torch.load(state_path)
        print("load_pretrain_model........")
        print(list(state.keys()))

        state_dict = state['state_dict']
        resnet50_state_dict = deepcopy(state_dict)
        for k in list(resnet50_state_dict.keys()):
            # only keep query encoder parameters; discard the fc projection head
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                resnet50_state_dict[k[len("module.encoder_q."):]] = resnet50_state_dict[k]

            # delete renamed or unused k
            del resnet50_state_dict[k]
       
        unet_state_dict = deepcopy(resnet50_state_dict)
        for k in list(unet_state_dict.keys()):
            unet_state_dict['encoder.' + k] = unet_state_dict[k]
            del unet_state_dict[k]
        print("unet_state_dict.......")
        self.unet.load_state_dict(unet_state_dict, strict=False)

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        # load pre-train from other data
        self.load_pretrain_model()
        unet_path = ''
       

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0
            for epoch in range(self.num_epochs):
                start = time.time()

                self.unet.train(True)
                epoch_loss = 0

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0

                print("train start......")

                for i, (images, GT, shape, _) in enumerate(tqdm(self.train_loader)):
                   

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    

                  
                    feat1, pred1, (SR1, SR2) = self.unet(images)

                    los = self.unet.module.regular(sampler=self.sampler, lab=GT, feat1=feat1, pred1=pred1, fov=None)
                   
                    SR1_probs = F.sigmoid(SR1)
                    SR2_probs = F.sigmoid(SR2)
                    
                    SR1_flat = SR1.view(SR1.size(0), -1)
                    SR2_flat = SR2.view(SR2.size(0), -1)
                   

                    GT_flat = GT.view(GT.size(0), -1)
                   

                    shape_flat = shape.view(shape.size(0), -1)
                                       loss1 = self.criterion(SR1_flat, GT_flat, shape_flat)
                    loss2 = self.criterion(SR2_flat, GT_flat, shape_flat)
                  
                    loss = los*0.1 + loss1*0.5 + loss2*0.5
    
                 
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    acc += get_accuracy(SR2_probs, GT)
                    SE += get_sensitivity(SR2_probs, GT)
                    SP += get_specificity(SR2_probs, GT)
                    PC += get_precision(SR2_probs, GT)
                    F1 += get_F1(SR2_probs, GT)
                    JS += get_JS(SR2_probs, GT)
                    DC += get_DC(SR2_probs, GT)
                    length = length + 1

                print('Training epoch {} times is {}: '.format(str(epoch), time.time() - start))

                # for each epoch, print a score
                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length
               
                print(
                    'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                        epoch + 1, self.num_epochs, \
                        epoch_loss, \
                        acc, SE, SP, PC, F1, JS, DC))

                # Decay learning rate
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))

                # ===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0

                print("val start.........")
                for i, (images, GT, shape, _) in enumerate(tqdm(self.valid_loader)):
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    _, _, SR = self.unet(images)
                    SR = F.sigmoid(SR)
                    acc += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)

                    length = length + 1
                # validation scores
                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length
                unet_score = F1

                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC))

                torchvision.utils.save_image(images.data.cpu(),
                                             os.path.join(self.result_path,
                                                          '%s_valid_%d_image.png' % (self.model_type, epoch + 1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                             os.path.join(self.result_path,
                                                          '%s_valid_%d_SR.png' % (self.model_type, epoch + 1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                             os.path.join(self.result_path,
                                                          '%s_valid_%d_GT.png' % (self.model_type, epoch + 1)))

        

                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                    unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f-best.pkl' % (
                    self.model_type, best_epoch, self.lr, self.num_epochs_decay, self.augmentation_prob))
                    torch.save(best_unet, unet_path)

                unet_path = os.path.join(self.model_path, '%s-%d.pkl' % (self.model_type, epoch))

                new_unet = self.unet.state_dict()
                torch.save(new_unet, unet_path)

                f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow(
                    [self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, self.num_epochs, self.num_epochs_decay,
                     self.augmentation_prob])
                f.close()

    def test(self):
        unet_path = os.path.join(self.model_path, '**.pkl')
        save_path = self.prediction_path + self.model_type + '/' + '**.pkl/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.build_model()
        self.unet.load_state_dict(torch.load(unet_path))

        self.unet.train(False)
        self.unet.eval()

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        length = 0
        for i, (images, GT, filename) in enumerate(self.test_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = F.sigmoid(self.unet(images))
            acc += get_accuracy(SR, GT)
            SE += get_sensitivity(SR, GT)
            SP += get_specificity(SR, GT)
            PC += get_precision(SR, GT)
            F1 += get_F1(SR, GT)
            JS += get_JS(SR, GT)
            DC += get_DC(SR, GT)

            # ========save prediction=========#
            out = SR.cpu().detach().numpy()
       
            for k in range(len(filename)):
                if not os.path.exists(save_path + filename[k].split('/')[0]):
                    os.makedirs(save_path + filename[k].split('/')[0])
                cv2.imwrite(save_path + filename[k], out[k, 0, :, :] * 255)

            length += images.size(0)

        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        unet_score = JS + DC

        f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([self.model_type, 'acc', 'SE', 'SP', 'PC', 'F1', 'JS', 'DC'])
        wr.writerow([self.model_type, acc, SE, SP, PC, F1, JS, DC])

        f.close()
