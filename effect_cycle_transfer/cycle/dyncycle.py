import os
import torch
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import itertools

from .models import S2S, SDmodel, AGmodel, ADmodel, Fengine, Iengine
from .utils import ImagePool, GANLoss
from .crosspolicy import CrossPolicy


class CycleGANModel():
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.istrain
        self.Tensor = torch.cuda.FloatTensor

        # 1,A -> source, 2,B -> target

        # load the pre-trained policy in the source domain, while the eval env is the target env
        self.cross_policy = CrossPolicy(opt)
        # state mapping: target -> source
        self.netG_2to1 = S2S(opt, dir='2to1').cuda()
        # state mapping: source -> target
        self.netG_1to2 = S2S(opt, dir='1to2').cuda()
        # action mapping: source -> target
        self.net_action_G_1to2 = AGmodel(opt, '1to2').cuda()  # map: source state X action -> target action

        # action mapping: target -> source
        self.net_action_G_2to1 = AGmodel(opt, '2to1').cuda()

        # target inverse dynamics model
        self.iengine = Iengine(opt, to_type='target')  # y_t x y_t+1 -> u_t
        self.netI_2 = self.iengine.imodel.cuda()

        self.iengine_1 = Iengine(opt, to_type='source')
        self.netI_1 = self.iengine_1.imodel.cuda()

        self.reset_buffer()
        # source state discriminator
        self.netD_1 = SDmodel(opt, dir='2to1').cuda()
        self.netD_2 = SDmodel(opt, dir='1to2').cuda()

        self.fake_A_pool = ImagePool(pool_size=128)
        self.fake_B_pool = ImagePool(pool_size=128)
        self.fake_action_A_pool = ImagePool(pool_size=128)
        self.fake_action_B_pool = ImagePool(pool_size=128)
        # define loss functions
        self.criterionGAN = GANLoss(tensor=self.Tensor).cuda()
        if opt.loss == 'l1':
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
        elif opt.loss == 'l2':
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionIdt = torch.nn.MSELoss()
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam([{'params': self.netI_2.parameters(), 'lr': 0.0},
                                             {'params': self.netI_1.parameters(), 'lr': 0.0},
                                             {'params': self.net_action_G_1to2.parameters(), 'lr': opt.lr_Ax},
                                             {'params': self.netG_1to2.parameters(), 'lr': opt.lr_Gx},
                                             {'params': self.netG_2to1.parameters(), 'lr': opt.lr_Gx}])
        self.optimizer_D_1 = torch.optim.Adam(self.netD_1.parameters())
        self.optimizer_D_2 = torch.optim.Adam(self.netD_2.parameters())

        self.init_start = True

        print('-----------------------------------------------')
        print('---------- Networks initialized ---------------')
        print('-----------------------------------------------')

    def update(self, opt):
        self.init_start = opt.init_start
        self.net_action_G_1to2.init_start = opt.init_start
        self.optimizer_G = torch.optim.Adam([{'params': self.netI_2.parameters(), 'lr': 0.0},
                                             {'params': self.netI_1.parameters(), 'lr': 0.0},
                                             {'params': self.net_action_G_1to2.parameters(), 'lr': opt.lr_Ax},
                                             {'params': self.netG_1to2.parameters(), 'lr': opt.lr_Gx},
                                             {'params': self.netG_2to1.parameters(), 'lr': opt.lr_Gx}])
        print('\n-----------------------------------------------')
        print('------------ model phase updated! -------------')
        print('-----------------------------------------------\n')

    def set_input(self, input):
        data1, data2 = input
        self.input_At0 = self.Tensor(data1[0])
        self.action_A = self.Tensor(data1[1])
        self.input_At1 = self.Tensor(data1[2])

        self.input_Bt0 = self.Tensor(data2[0])
        self.action_B = self.Tensor(data2[1])
        self.input_Bt1 = self.Tensor(data2[2])

    def forward(self):
        self.real_At0 = Variable(self.input_At0)
        self.action_A = Variable(self.action_A)
        self.real_At1 = Variable(self.input_At1)

        self.real_Bt0 = Variable(self.input_Bt0)
        self.action_B = Variable(self.action_B)
        self.real_Bt1 = Variable(self.input_Bt1)

    def test(self):
        # forward
        self.forward()
        # G_A and G_B
        self.backward_G()
        self.backward_D_B()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * self.opt.lambda_D
        # backward
        if self.isTrain:
            loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_A = self.fake_A_pool.query(self.fake_At0)
        loss_D_1 = self.backward_D_basic(self.netD_1, self.real_At0, fake_A)
        self.loss_D_1 = loss_D_1.item()

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_Bt0)
        loss_D_2 = self.backward_D_basic(self.netD_2, self.real_Bt0, fake_B)
        self.loss_D_2 = loss_D_2.item()

    def backward_G(self):
        lambda_G_B0 = self.opt.lambda_G0
        lambda_G_A0 = self.opt.lambda_G0
        # lambda_G_cycle = self.opt.lambda_G_cyc
        # lambda_cycle_action = self.opt.lambda_action_cyc
        lambda_F = self.opt.lambda_F

        # ***************************
        #       state cycle part
        # ***************************

        # generator target state -> source state loss
        fake_At0 = self.netG_2to1(self.real_Bt0)
        pred_fake = self.netD_1(fake_At0)
        loss_G_Bt0 = self.criterionGAN(pred_fake, True) * lambda_G_B0

        # generator source state -> target state loss
        fake_Bt0 = self.netG_1to2(self.real_At0)
        pred_fake = self.netD_2(fake_Bt0)
        loss_G_At0 = self.criterionGAN(pred_fake, True) * lambda_G_A0

        # cycle consistency part
        cycle_fake_At0 = self.netG_1to2(fake_At0)
        cycle_label_2 = torch.zeros_like(self.real_Bt0).float().cuda()

        cycle_fake_Bt0 = self.netG_2to1(fake_Bt0)
        cycle_label_1 = torch.zeros_like(self.real_At0).float().cuda()
        loss_cycle = (self.criterionCycle(cycle_fake_At0 - self.real_Bt0, cycle_label_2) +
                      self.criterionCycle(cycle_fake_Bt0 - self.real_At0, cycle_label_1)) * lambda_F

        # ***************************
        #       action part
        # ***************************
        fake_target_t0 = self.netG_1to2(self.real_At0)
        fake_target_t1 = self.netG_1to2(self.real_At1)
        target_action_2 = self.netI_2(fake_target_t0, fake_target_t1)

        pred_fake_target_action_2 = self.net_action_G_1to2(self.real_At0, self.action_A)
        label_2 = torch.zeros_like(pred_fake_target_action_2).float().cuda()
        loss_action_2 = self.criterionCycle(target_action_2 - pred_fake_target_action_2, label_2) * lambda_F

        fake_source_t0 = self.netG_2to1(self.real_Bt0)
        fake_source_t1 = self.netG_2to1(self.real_Bt1)
        target_action_1 = self.netI_1(fake_source_t0, fake_source_t1)

        pred_fake_target_action_1 = self.net_action_G_2to1(self.real_Bt0, self.action_B)
        label_1 = torch.zeros_like(pred_fake_target_action_1).float().cuda()
        loss_action_1 = self.criterionCycle(target_action_1 - pred_fake_target_action_1, label_1) * lambda_F

        # ***************************
        #       loss backward part
        # ***************************

        # combined loss
        loss_action = loss_action_1 + loss_action_2
        loss_G = loss_G_Bt0 + loss_G_At0 + loss_cycle
        loss = loss_G + loss_action

        if self.isTrain:
            loss.backward()

        # ***************************
        #      postprocess part
        # ***************************
        # get the data for training discriminator

        self.fake_At0 = fake_At0.data
        self.fake_Bt0 = fake_Bt0.data

        self.loss_G_B = loss_G_Bt0.item()
        self.loss_G_A = loss_G_At0.item()
        self.loss_cycle = loss_cycle.item()
        self.loss_action_effect = loss_action.item()

        self.loss_state_lt0, self.loss_state_lt1 = 0, 0

        self.gt0 = self.real_Bt0
        self.gt1 = self.real_Bt1
        self.gt_buffer.append(self.gt0.cpu().data.numpy())
        self.pred_buffer.append(self.fake_At0.cpu().data.numpy())

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_B
        self.optimizer_D_2.zero_grad()
        self.backward_D_B()
        self.optimizer_D_2.step()
        # D_A
        self.optimizer_D_1.zero_grad()
        self.backward_D_A()
        self.optimizer_D_1.step()

    def fetch(self):
        real_B = (self.real_Bt0.cpu().data.numpy(),
                  self.action_B.cpu().data.numpy(),
                  self.real_Bt1.cpu().data.numpy())
        fake_B = (self.fake_At0.cpu().data.numpy(),
                  self.action_B.cpu().data.numpy(),
                  self.fake_At1.cpu().data.numpy())
        return (real_B, fake_B)

    def get_current_errors(self):
        ret_errors = OrderedDict([
            ('G_B2A', self.loss_G_B),
            ('G_A2B', self.loss_G_A),
            ('cycle', self.loss_cycle),
            ('action_effect', self.loss_action_effect),
            ('D_A', self.loss_D_1),
            ('D_B', self.loss_D_2)
        ])
        return ret_errors

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, path):
        save_filename = 'model_{}.pth'.format(network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    def save(self, path):
        self.save_network(self.netG_2to1, 'G_B2A', path)
        self.save_network(self.netG_1to2, 'G_A2B', path)
        self.save_network(self.net_action_G_1to2, 'G_act_A2B', path)
        self.save_network(self.net_action_G_2to1, 'G_act_B2A', path)
        self.save_network(self.netD_1, 'D_A', path)
        self.save_network(self.netD_2, 'D_B', path)

    def load_network(self, network, network_label, path):
        weight_filename = 'model_{}.pth'.format(network_label)
        weight_path = os.path.join(path, weight_filename)
        network.load_state_dict(torch.load(weight_path))

    def load(self, path):
        self.load_network(self.netG_2to1, 'G_B2A', path)
        self.load_network(self.netG_1to2, 'G_A2B', path)
        self.load_network(self.net_action_G_1to2, 'G_act_A2B', path)
        self.load_network(self.net_action_G_2to1, 'G_act_B2A', path)
        self.load_network(self.netD_1, 'D_A', path)
        self.load_network(self.netD_2, 'D_B', path)

    def reset_buffer(self):
        self.gt_buffer = []
        self.pred_buffer = []
        self.gt_buffer1 = []
        self.pred_buffer1 = []

    def show_points(self, gt_data, pred_data):
        ncols = int(np.sqrt(gt_data.shape[1])) + 1
        nrows = int(np.sqrt(gt_data.shape[1])) + 1
        assert (ncols * nrows >= gt_data.shape[1])
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i >= gt_data.shape[1]:
                continue
            ax.scatter(gt_data[:, ax_i], pred_data[:, ax_i], s=3, label='xyz_{}'.format(ax_i))

    def npdata(self, item):
        return item.cpu().data.numpy()

    def visual(self, path=None):
        # gt_data = np.vstack(self.gt_buffer)
        # pred_data = np.vstack(self.pred_buffer)
        # self.show_points(gt_data,pred_data)
        # plt.savefig(path)
        # plt.cla()
        # plt.clf()

        # gt_data = np.vstack(self.gt_buffer1)
        # pred_data = np.vstack(self.pred_buffer1)
        # print(list(abs(gt_data - pred_data).mean(0)), abs(gt_data - pred_data).mean())
        self.reset_buffer()
