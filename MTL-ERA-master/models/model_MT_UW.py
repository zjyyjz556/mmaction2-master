from models.tcn import TemporalENCDECConvNet
from models.gcn import *
import torch.nn.functional as F
import yaml

#以下是自己加的包
from models.ctrgcn import *
from tools.train import *

try:
    with open('./config_files/config_UW_data.yml', 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config_data file')
num_class = config['NUMBER_OF_CLASSES']


def weightinit(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('Linear') != -1:
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.xavier_normal_(m.weight)
    if classname.find('LSTM') != -1:
        torch.nn.init.xavier_normal_(m.weight_ih_l0)
        torch.nn.init.orthogonal_(m.weight_hh_l0)
        torch.nn.init.constant_(m.bias_ih_l0, 0)
        torch.nn.init.constant_(m.bias_hh_l0, 0)


########################################################
# Multil task
class gcnEdtcnREBA_tanh(nn.Module):
    def __init__(self, hidden=25, kernel_size=8):
        super(gcnEdtcnREBA_tanh, self).__init__()
        self.GCN = GCNModel(in_channels=3)  # (N, 3, T, 15, M) -> (N,  T, 3840)
        self.adaptpool = nn.AdaptiveAvgPool1d(2048)  # (N,  T, 3840) -> (N,  T, 2048)
        self.tcn = TemporalENCDECConvNet(2048, hidden, kernel_size=kernel_size,
                                         dropout=0.005)  # (N, 1024, T) -> (N, 1024, T)
        self.ln1 = nn.Linear(2048, 1024)  # (N,  T, 2048) -> (N, T, 1024)
        self.l_classification = nn.Linear(1024, num_class)  # (N,  T, 1024) -> (N, T, 17)
        self.ln2 = nn.Linear(2048, 256)  # (N,  T, 2048) -> (N, T, 256)
        self.LSTM1 = nn.LSTM(256, 1024, num_layers=3, batch_first=True)

        # self.ln2 = nn.Linear(2048 + 17, 1024)  # (N,  T, 2075) -> (N, T, 1024)
        self.l_regression = nn.Linear(1024, 1)  # (N,  T, 1024) -> (N, T, 1)

    def forward(self, x):
        # xx = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).unsqueeze(4)
        x = self.adaptpool(self.GCN(x))
        x_class = self.tcn(torch.relu(x.permute(0, 2, 1)))
        classification_score = self.l_classification(torch.tanh(self.ln1(x_class.permute(0, 2, 1))))
        x_reg, _ = self.LSTM1(torch.tanh(self.ln2(x)))
        regression_output = self.l_regression(x_reg)
        return classification_score, regression_output


class gcnEdtcnREBA_emb(nn.Module):
    def __init__(self, hidden=25, kernel_size=8):
        super(gcnEdtcnREBA_emb, self).__init__()
        self.TDCNN = build_model(dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=60,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(),
    test_cfg=dict(average_clips='prob')), train_cfg=None, test_cfg=None)
#        self.GCN = GCNModel(in_channels=3)  # (N, 3, T, 15, M) -> (N,  T, 3840)
#        model_args = {'num_class': 17, 'num_point': 15, 'num_person': 1, 'graph': 'graph.ucla.Graph', 'graph_args': {'labeling_mode': 'spatial'}}
#        self.ctrgcn = Model(**model_args)
        self.adaptpool = nn.AdaptiveAvgPool1d(2048)  # (N,  T, 3840) -> (N,  T, 1024)
        self.tcn = TemporalENCDECConvNet(2048, hidden, kernel_size=kernel_size,
                                         dropout=0.005)  # (N, 1024, T) -> (N, 1024, T)
        self.ln1 = nn.Linear(2048, 1024)  # (N,  T, 2048) -> (N, T, 1024)
        self.l_classification = nn.Linear(1024, num_class)  # (N,  T, 1024) -> (N, T, 17)
        self.ln2 = nn.Linear(2048 + num_class, 256)  # (N,  T, 2048) -> (N, T, 256)
        self.LSTM1 = nn.LSTM(256, 1024, num_layers=3, batch_first=True)  # (N,  T, 256) -> (N, T, 1024)

        self.l_regression = nn.Linear(1024, 1)  # (N,  T, 1024) -> (N, T, 1)

    def forward(self, x):
        N, T, V, C = x.size()
#        y = self.ctrgcn(x)
#        x = x.permute(0, 3, 1, 2).unsqueeze(4)
#        x = self.adaptpool(self.GCN(x))
#        x = self.adaptpool(self.ctrgcn(x))
        x = self.adaptpool(self.TDCNN(x))
        x_class = self.tcn(torch.relu(x.permute(0, 2, 1)))
        classification_score = self.l_classification(F.relu(self.ln1(x_class.permute(0, 2, 1))))
        x_reg, _ = self.LSTM1(torch.tanh(self.ln2(torch.cat([x, F.softmax(classification_score)], dim=2))))
        regression_output = self.l_regression(x_reg)

        return classification_score, regression_output


########################################################
# Single task
class gcn_reg(nn.Module):
    def __init__(self, hidden=25, kernel_size=8):
        super(gcn_reg, self).__init__()
        self.GCN = GCNModel(in_channels=3)  # (N, 3, T, 15, M) -> (N,  T, 3840)
        self.adaptpool = nn.AdaptiveAvgPool1d(2048)  # (N,  T, 3840) -> (N,  T, 1024)
        self.tcn = TemporalENCDECConvNet(2048, hidden, kernel_size=kernel_size,
                                         dropout=0.005)  # (N, 1024, T) -> (N, 1024, T)
        self.ln2 = nn.Linear(2048, 512)  # (N,  T, 2048) -> (N, T, 256)
        self.LSTM1 = nn.LSTM(512, 1024, num_layers=4, batch_first=True)
        self.l_regression = nn.Linear(1024, 1)  # (N,  T, 1024) -> (N, T, 1)

    def forward(self, x):
        # xx = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).unsqueeze(4)
        x = self.adaptpool(self.GCN(x))
        x_reg, _ = self.LSTM1(torch.relu(self.ln2(x)))
        regression_output = self.l_regression(x_reg)
        return regression_output, 0  # , regression_output


class gcnEdtcn_class(nn.Module):
    def __init__(self, hidden=25, kernel_size=8):
        super(gcnEdtcn_class, self).__init__()
        self.GCN = GCNModel(in_channels=3)  # (N, 3, T, 15, M) -> (N,  T, 3840)
        self.adaptpool = nn.AdaptiveAvgPool1d(2048)  # (N,  T, 3840) -> (N,  T, 1024)
        self.tcn = TemporalENCDECConvNet(2048, hidden, kernel_size=kernel_size,
                                         dropout=0.005)  # (N, 1024, T) -> (N, 1024, T)
        self.ln1 = nn.Linear(2048, 1024)  # (N,  T, 2048) -> (N, T, 1024)
        self.l_classification = nn.Linear(1024, num_class)  # (N,  T, 1024) -> (N, T, 17)

    def forward(self, x):
        # xx = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).unsqueeze(4)
        x = self.adaptpool(self.GCN(x))
        x_class = self.tcn(x.permute(0, 2, 1))
        classification_score = self.l_classification(torch.relu(self.ln1(x_class.permute(0, 2, 1))))
        return classification_score, 0  # , classification_output
