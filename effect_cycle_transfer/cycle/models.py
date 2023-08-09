import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn


class S2S(nn.Module):
    def __init__(self, opt, dir='1to2'):
        super(S2S, self).__init__()
        if dir == '2to1':
            self.state_dim1 = opt.state_dim1
            self.state_dim2 = opt.state_dim2
        else:
            self.state_dim1 = opt.state_dim2
            self.state_dim2 = opt.state_dim1
        self.ssfc = nn.Sequential(
            nn.Linear(self.state_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim1)
        )

    def forward(self, state):
        return self.ssfc(state)


class SDmodel(nn.Module):
    def __init__(self, opt, dir='2to1'):
        super(SDmodel, self).__init__()
        if dir == '2to1':
            self.state_dim1 = opt.state_dim1
        else:
            self.state_dim1 = opt.state_dim2
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, state):
        return self.fc(state)


class AGmodel(nn.Module):
    def __init__(self, opt, dir='1to2'):
        super(AGmodel, self).__init__()
        self.dir = dir
        self.opt = opt
        self.env = opt.env
        if self.dir == '1to2':
            self.state_dim = opt.state_dim1
            self.action_dim1 = opt.action_dim1
            self.action_dim2 = opt.action_dim2
        elif self.dir == '2to1':
            self.state_dim = opt.state_dim2
            self.action_dim1 = opt.action_dim2
            self.action_dim2 = opt.action_dim1
        self.init_start = opt.init_start
        self.statefc = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.actionfc = nn.Sequential(
            nn.Linear(self.action_dim1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.output = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim2),
            nn.Tanh()
        )
        self.max_action = 1.0

    def forward(self, state, action):
        if self.init_start:
            new_action = self.get_init_action(action)
        else:
            if state.ndim == 1:
                state = state[None, :]
            state_post = self.statefc(state)
            action_post = self.actionfc(action)
            state_action_post = torch.cat([state_post, action_post], dim=1)
            new_action = self.output(state_action_post) * self.max_action
        return new_action

    def get_init_action(self, action):
        """the action should be initialized, directly cloned from the nearest joint.
        This handcraft is determined by the construction method of new morphology agent."""
        # TODO check this trick
        if self.env == 'Swimmer-v2':
            # 3part -> 4part: 0 1 => 0 1 0
            # 4part -> 3part: 0 1 2 => 0 1
            if self.dir == '1to2':
                new_action = torch.cat((action, action[:, 0:1]), 1)
            else:
                new_action = action[:, :2]
        elif self.env == 'HalfCheetah-v2':
            # 3leg -> 4leg: 0 1 2 3 4 5 => 0 1 2 0 1 2 3 4 5
            # 4leg -> 3leg: 0 1 2 3 4 5 6 7 8 => 0 1 2 6 7 8
            if self.dir == '1to2':
                new_action = torch.cat((action[:, :3], action[:, :3], action[:, 3:6]), 1)
            else:
                new_action = torch.cat((action[:, :3], action[:, 6:9]), 1)
        else:
            new_action = action
        return new_action


class ADmodel(nn.Module):
    def __init__(self, opt, dir='1to2'):
        super(ADmodel, self).__init__()
        self.dir = dir
        if self.dir == '1to2':
            self.action_dim1 = opt.action_dim1
            self.action_dim2 = opt.action_dim2
        elif self.dir == '2to1':
            self.action_dim1 = opt.action_dim2
            self.action_dim2 = opt.action_dim1
        self.init_start = opt.init_start
        self.fc = nn.Sequential(
            nn.Linear(self.action_dim2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, action):
        return self.fc(action)


class Fmodel(nn.Module):
    def __init__(self, opt):
        super(Fmodel, self).__init__()
        self.state_dim = opt.state_dim1
        self.action_dim = opt.action_dim1
        self.statefc = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.actionfc = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.predfc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.state_dim)
        )

    def forward(self, state, action):
        state_feature = self.statefc(state)
        action_feature = self.actionfc(action)
        feature = torch.cat((state_feature, action_feature), 1)
        return self.predfc(feature)


class Imodel(nn.Module):
    def __init__(self, opt, to_type='target'):
        super(Imodel, self).__init__()
        if to_type == 'target':
            self.state_dim = opt.state_dim2
            self.action_dim = opt.action_dim2
        else:
            self.state_dim = opt.state_dim1
            self.action_dim = opt.action_dim1
        self.statefc = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.predfc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )

    def forward(self, now_state, next_state):
        now_state_feature = self.statefc(now_state)
        next_state_feature = self.statefc(next_state)
        feature = torch.cat((now_state_feature, next_state_feature), 1)
        return self.predfc(feature)


class Fengine:
    def __init__(self, opt):
        self.fmodel = Fmodel(opt).cuda()
        self.opt = opt

    def train_statef(self, dataset):
        self.env_logs = os.path.join(self.opt.log_root, 'data/{}_data'.format(self.opt.env))
        self.data_root1 = os.path.join(self.env_logs, '{}'.format(self.opt.data_id1))
        weight_path = os.path.join(self.data_root1, 'forward.pth')
        if self.opt.pretrain_f:
            self.fmodel.load_state_dict(torch.load(weight_path))
            return None
        lr = 1e-3
        optimizer = torch.optim.Adam(self.fmodel.parameters(), lr=lr)
        loss_fn = nn.L1Loss()
        now, act, nxt = dataset
        batch_size = 32
        data_size = int(now.shape[0] / batch_size)
        for epoch in range(10):
            if epoch in [3, 7, 10, 15]:
                lr *= 0.5
                optimizer = torch.optim.Adam(self.fmodel.parameters(), lr=lr)
            epoch_loss, cmp_loss = 0, 0
            idx = list(range(now.shape[0]))
            random.shuffle(idx)
            now = now[idx]
            act = act[idx]
            nxt = nxt[idx]
            for i in range(data_size):
                start = i * batch_size
                end = start + batch_size
                state = torch.tensor(now[start:end]).float().cuda()
                action = torch.tensor(act[start:end]).float().cuda()
                result = torch.tensor(nxt[start:end]).float().cuda()

                out = self.fmodel(state, action)
                loss = loss_fn(out, result)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                cmp_loss += loss_fn(state, result).item()

            torch.save(self.fmodel.state_dict(), weight_path)
        print('source domain dynamics model training loss:{:.7f} cmp:{:.7f}'.format(epoch_loss / data_size,
                                                                                    cmp_loss / data_size))


class Iengine:
    def __init__(self, opt, to_type='target'):
        self.imodel = Imodel(opt, to_type=to_type).cuda()
        self.to_type = to_type
        self.opt = opt

    def train_statef(self, dataset):
        if self.to_type == 'target':
            self.env_logs = os.path.join(self.opt.log_root, 'data/{}_data'.format(self.opt.target_env))
            self.data_root2 = os.path.join(self.env_logs, '{}'.format(self.opt.data_id2))
            weight_path = os.path.join(self.data_root2, 'inverse.pth')
        else:
            self.env_logs = os.path.join(self.opt.log_root, 'data/{}_data'.format(self.opt.env))
            self.data_root2 = os.path.join(self.env_logs, '{}'.format(self.opt.data_id1))
            weight_path = os.path.join(self.data_root2, 'inverse.pth')

        if self.opt.pretrain_i:
            self.imodel.load_state_dict(torch.load(weight_path))
            print('load the pretrained inverse dynamics model in the target domain')
            return None
        lr = 1e-3
        optimizer = torch.optim.Adam(self.imodel.parameters(), lr=lr)
        loss_fn = nn.L1Loss()
        now, act, nxt = dataset
        batch_size = 32
        data_size = int(now.shape[0] / batch_size)
        for epoch in range(10):
            if epoch in [3, 7, 10, 15]:
                lr *= 0.5
                optimizer = torch.optim.Adam(self.imodel.parameters(), lr=lr)
            epoch_loss = 0
            idx = list(range(now.shape[0]))
            random.shuffle(idx)
            now = now[idx]
            act = act[idx]
            nxt = nxt[idx]
            for i in tqdm(range(data_size)):
                start = i * batch_size
                end = start + batch_size
                state = torch.tensor(now[start:end]).float().cuda()
                result = torch.tensor(act[start:end]).float().cuda()
                next_state = torch.tensor(nxt[start:end]).float().cuda()

                out = self.imodel(state, next_state)
                loss = loss_fn(out, result)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            torch.save(self.imodel.state_dict(), weight_path)
        print('{} domain inverse dynamics model training loss:{:.7f} '.format(self.to_type, epoch_loss / data_size
                                                                              ))
