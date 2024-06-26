import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import torch
import cv2
from network import U_Net_cl
from U2net import U2NET
from torch import optim
import torch.nn.functional as F
from evaluation import *
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def similar_matrix1(q, k, temperature=0.1): 
    qfh, qfl, qfall, qbh, qbl, qball = torch.chunk(q, 6, dim=0)
    kfh, kfl, kfall, kbh, kbl, kball = torch.chunk(k, 6, dim=0)
    # negative logits: NxK
    l_pos = torch.einsum('nc,kc->nk', [qfall, kball])
    l_neg = torch.einsum('nc,kc->nk', [kfall, qball])
    
    return l_pos.mean() + l_neg.mean()

def similar_matrix2(q, k, temperature=0.1):  
    qfh, qfl, qfall, qbh, qbl, qball = torch.chunk(q, 6, dim=0)
    kfh, kfl, kfall, kbh, kbl, kball = torch.chunk(k, 6, dim=0)
    # negative logits: NxK
    l_pos = torch.einsum('nc,kc->nk', [qfl, kfh])
    l_neg = torch.einsum('nc,kc->nk', [qbl, kbh])
    
    return 1 - l_pos.mean() - l_neg.mean()


class MlpNorm(nn.Module):
    def __init__(self, dim_inp=256, dim_out=64):
        super(MlpNorm, self).__init__()
        dim_mid = min(dim_inp, dim_out) 
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
        
        self.temperature = temperature
        self.proj_num_layers = proj_num_layers
        self.pred_num_layers = pred_num_layers

        self.projector = MlpNorm(32, 128)
        self.predictor = MlpNorm(128, 128)

    def forward(self, img, **args): 
        out = self.encoder(img, **args)
        self.pred = self.encoder.pred
        self.feat = self.encoder.feat
        print(self.feat.shape, self.pred.shape)
        return out 

    def regular(self, sampler, lab, fov=None):  # contrastive loss split by classification
        feat = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
        print(self.feat.shape, self.pred.shape)
        print(feat.shape)
        proj = self.projector(feat)
        self.proj = proj
        pred = self.predictor(proj)
        
        losSG1 = self.loss1(pred, proj.detach(), temperature=self.temperature)
        losSG2 = self.loss1(proj, pred.detach(), temperature=self.temperature)
    

        
        losSG3 = self.loss2(proj, pred.detach(), temperature=self.temperature)
        losSG4 = self.loss2(proj, pred.detach(), temperature=self.temperature)
       
        return (losSG1 + losSG2) + (losSG3 + losSG4)


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
       
        
        focal_weight = (self.weight * target + (1 - self.weight) * (1 - target)) * pt.pow(self.gamma) * (
                    1 + shapes.to(device))
        loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='none') * focal_weight
        print(loss.mean())
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



def options():
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--model_type', type=str, default='ECSNet', help=' ')
    parser.add_argument('--model_name', type=str, default='**.pkl', help=' ')
    parser.add_argument('--prediction_path', type=str, default='./')
    parser.add_argument('--model_path', type=str, default='./')
    parser.add_argument('--test_path', type=str, default='./')
    parser.add_argument('--result_path', type=str, default='./result/')

    config = parser.parse_args()

    return config

class Solver1(object):
    def __init__(self, config, test_loader):

        # Data loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
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
        self.model_name = config.model_name
        self.result_path = config.result_path
        self.prediction_path = config.prediction_path


        self.device = torch.device('cuda')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

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
        print(self.device)

        self.unet = torch.nn.DataParallel(self.unet)
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
            x = x.cuda()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()


    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img

    def test(self):
        unet_path = os.path.join(self.model_path, self.model_name)
        save_path = self.prediction_path + self.model_type + '/' + self.model_name.split('.')[0] +'/'
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
        dice = 0
        iou = 0
        hd = 0

        for i, (images, GT, filename) in enumerate(self.test_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = F.sigmoid(self.unet(images)).to(self.device)

            acc += get_accuracy(SR, GT)
            SE += get_sensitivity(SR, GT)
            SP += get_specificity(SR, GT)
            PC += get_precision(SR, GT)
            F1 += get_F1(SR, GT)
            JS += get_JS(SR, GT)
            #DC += get_DC(SR, GT)
            #dice += get_DICE(SR, GT)
            from medpy.metric import binary

            

            iou += get_IOU(SR, GT)
            


            #========save prediction=========#
            out = SR.cpu().detach().numpy()

            for k in range(len(filename)):
                if not os.path.exists(save_path+filename[k].split('/')[0]):
                    os.makedirs(save_path+filename[k].split('/')[0])
                cv2.imwrite(save_path+filename[k],out[k,0,:,:]*255)
                if get_F1(SR, GT) < 0.7:
                    print(filename[k],get_F1(SR, GT))

            length = length + 1


        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        dice = dice / length
        iou = iou/length
        #hd = hd/length

        print(acc,SE,SP,PC,F1,JS,DC,dice,iou)




if __name__ == '__main__':
    config = options()
    cudnn.benchmark = True

    lr = random.random() * 0.0005 + 0.0000005
    augmentation_prob = random.random() * 0.7
    epoch = random.choice([100, 150, 200, 250])
    decay_ratio = random.random() * 0.8
    decay_epoch = int(epoch * decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    #==========  load test data  =====#
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.4)


    solver = Solver1(config, test_loader)

    solver.test()

