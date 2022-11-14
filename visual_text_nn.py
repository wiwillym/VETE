import torch.nn as nn

class VTNN(nn.Module):
  def __init__(self, p=0.0, config_preset=1):
    super(VTNN, self).__init__()
    #self.flatten = nn.Flatten()
    if config_preset == 1:
      self.layers = nn.Sequential(
          nn.Linear(2048, 4000),
          nn.BatchNorm1d(4000),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(4000, 2000),
          nn.BatchNorm1d(2000),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2000, 3000),
          nn.BatchNorm1d(3000),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(3000, 2048),
          nn.BatchNorm1d(2048),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2048, 1024),
      )
    elif config_preset == 2:
      self.layers = nn.Sequential(
          nn.Linear(2048, 4000),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(4000, 2000),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2000, 3000),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(3000, 2048),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2048, 1024),
      )
    elif config_preset == 3:
      self.layers = nn.Sequential(
          nn.Linear(2048, 2048),
          nn.BatchNorm1d(2048),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 1024),
      )
    elif config_preset == 4:
      self.layers = nn.Sequential(
          nn.Linear(2048, 2048),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 1024),
      )
    elif config_preset == 5:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 2048),
          nn.BatchNorm1d(2048),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2048, 1024),
      )
    elif config_preset == 6:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 2048),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2048, 1024),
      )
    elif config_preset == 7:
      self.layers = nn.Sequential(
          nn.Linear(2048, 4096),
          nn.BatchNorm1d(4096),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(4096, 2048),
          nn.BatchNorm1d(2048),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2048, 1024),
      )
    elif config_preset == 8:
      self.layers = nn.Sequential(
          nn.Linear(2048, 4096),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(4096, 2048),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(2048, 1024),
      )
    elif config_preset == 9:
      self.layers = nn.Sequential(
          nn.Linear(2048, 4096),
          nn.BatchNorm1d(4096),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(4096, 1024),
      )
    elif config_preset == 10:
      self.layers = nn.Sequential(
          nn.Linear(2048, 4096),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(4096, 1024),
      )


  def forward(self, x):
    x = x.view(-1,2048)
    logits = self.layers(x)
    return logits

class VTNN8dim(nn.Module):
  def __init__(self, p=0.0, config_preset=1):
    super(VTNN8dim, self).__init__()
    #self.flatten = nn.Flatten()
    if config_preset == 1:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(256, 128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(128, 64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(64, 32),
          nn.BatchNorm1d(32),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(32, 16),
          nn.BatchNorm1d(16),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(16, 8),
      )
    elif config_preset == 2:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(32, 16),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(16, 8),
      )
    elif config_preset == 3:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(128, 64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(64, 16),
          nn.BatchNorm1d(16),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(16, 8),
      )
    elif config_preset == 4:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(64, 16),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(16, 8),
      )
    elif config_preset == 5:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(256, 64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(64, 8),
      )
    elif config_preset == 6:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 256),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(256, 64),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(64, 8),
      )
    elif config_preset == 7:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(128, 8),
      )
    elif config_preset == 8:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(128, 8),
      )
    elif config_preset == 9:
      self.layers = nn.Sequential(
          nn.Linear(2048, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(128, 8),
      )
    elif config_preset == 10:
      self.layers = nn.Sequential(
          nn.Linear(2048, 512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(128, 8),
      )


  def forward(self, x):
    x = x.view(-1,2048)
    logits = self.layers(x)
    return logits


class VTNN128dim(nn.Module):
  def __init__(self, p=0.0, config_preset=1):
    super(VTNN128dim, self).__init__()
    #self.flatten = nn.Flatten()
    if config_preset == 1:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(256, 128),
      )
    elif config_preset == 2:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(256, 128),
      )
    elif config_preset == 3:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
      )
    elif config_preset == 4:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
      )
    elif config_preset == 5:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(256, 128),
      )
    elif config_preset == 6:
      self.layers = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(1024, 256),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(256, 128),
      )
    elif config_preset == 7:
      self.layers = nn.Sequential(
          nn.Linear(2048, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
      )
    elif config_preset == 8:
      self.layers = nn.Sequential(
          nn.Linear(2048, 512),
          nn.ReLU(),
          nn.Dropout(p=p),
          nn.Linear(512, 128),
      )


  def forward(self, x):
    x = x.view(-1,2048)
    logits = self.layers(x)
    return logits


'''
class VTNN(nn.Module):
  def __init__(self, p=0.0, n_hidden=3):
    super(VTNN, self).__init__()
    self.flatten = nn.Flatten()
    if n_hidden == 1:
      self.layers = nn.Sequential(
          nn.Linear(2048, 4096),
          nn.ReLU(),
          nn.Linear(4096, 2048),
          nn.ReLU(),
          nn.Linear(2048, 1024),
          nn.Dropout(p=p),
        )
    elif n_hidden == 2:
      self.layers = nn.Sequential(
        nn.Linear(2048, 4000),
        nn.ReLU(),
        nn.Linear(4000, 3000),
        nn.ReLU(),
        nn.Linear(3000, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.Dropout(p=p),
      )
    elif n_hidden == 3:
      self.layers = nn.Sequential(
        nn.Linear(2048, 4000),
        nn.ReLU(),
        nn.Linear(4000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 3000),
        nn.ReLU(),
        nn.Linear(3000, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.Dropout(p=p),
      )
    else:
      self.layers = nn.Sequential(
          nn.Linear(2048, 2048),
          nn.BatchNorm1d(2048),
          nn.ReLU(),
          nn.Linear(2048, 1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          #nn.Linear(1024, 512),
          #nn.BatchNorm1d(512),
          #nn.ReLU(),
          nn.Linear(1024, 1024)
      )

  def forward(self, x):
    x = x.view(-1,2048)
    logits = self.layers(x)
    return logits
'''