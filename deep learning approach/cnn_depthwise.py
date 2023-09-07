
import torch
import torch.nn as nn


class GlobalAvgPool1D(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1D,self).__init__()
    def forward(self,x):
        return x.mean(axis=-1) 
    
    

class cnnlstm(nn.Module):
    def __init__(self):
        super().__init__()
       
        
        self.conv_current =nn.Sequential(nn.Conv1d(2, 16, kernel_size=3, stride = 2, padding=0),
                                     nn.BatchNorm1d(16),
                                 nn.ReLU(),
                                  nn.Conv1d(16, 32, kernel_size=3, stride = 2, padding=0),
                                     nn.BatchNorm1d(32),
                                 nn.ReLU(),
                                  nn.Conv1d(32, 64, kernel_size=2, stride = 2, padding=0),
                                     nn.BatchNorm1d(64),
                                 nn.ReLU())
        self.conv_vib =nn.Sequential(nn.Conv1d(2, 16, kernel_size=3, stride = 2, padding=0),
                                     nn.BatchNorm1d(16),
                                 nn.ReLU(),
                                  nn.Conv1d(16, 32, kernel_size=3, stride = 2, padding=0),
                                     nn.BatchNorm1d(32),
                                 nn.ReLU(),
                                  nn.Conv1d(32, 64, kernel_size=2, stride = 2, padding=0),
                                     nn.BatchNorm1d(64),
                                 nn.ReLU())
        self.conv_acous = nn.Sequential(nn.Conv1d(2, 16, kernel_size=3, stride =2, padding=0),
                                     nn.BatchNorm1d(16),
                                 nn.ReLU(),
                                  nn.Conv1d(16, 32, kernel_size=3, stride = 2, padding=0),
                                     nn.BatchNorm1d(32),
                                 nn.ReLU(),
                                  nn.Conv1d(32, 64, kernel_size=2, stride =2, padding=0),
                                     nn.BatchNorm1d(64),
                                 nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv1d(192, 256, kernel_size=3, stride = 2, padding=0),
                                     nn.BatchNorm1d(256),
                                 nn.ReLU(),
                                  nn.Conv1d(256, 512, kernel_size=3, stride = 2, padding=0),
                                     nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                  nn.Conv1d(512, 1024, kernel_size=3, stride = 2, padding=0),
                                     nn.BatchNorm1d(1024),
                                 nn.ReLU())
        
        self.gap = GlobalAvgPool1D()
        # self.lstm = nn.LSTM(
        #     input_size=7,
        #     hidden_size=5,
        #     num_layers=2,
        #     batch_first=True
        # )


        self.fc = nn.Sequential(
                                 nn.Linear(1024, 128),
           
                                 nn.Linear(128, 32),
                                 nn.Linear(32, 1),
                               )
        
        

    def forward(self, x):
        x1 = self.conv_current(x[:,0:2,:])
        x2 = self.conv_vib(x[:,2:4,:])
        x3 = self.conv_acous(x[:,4:6,:])

        feature=torch.cat((x1,x2,x3),dim=1)
        x4 = self.conv1(feature)
        hidden = self.gap(x4)
        x = self.fc(hidden)
        return x