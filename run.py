import numpy as np
import xarray as xr
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os

from collections import OrderedDict
from pathlib import Path

from DataGenerator import DataGenerator
from pred import RNN

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


z = xr.open_mfdataset('../geopotential/*.nc', combine='by_coords')
pv = xr.open_mfdataset('../potential_vorticity/*.nc', combine='by_coords')
r = xr.open_mfdataset('../relative_humidity/*.nc', combine='by_coords')
q = xr.open_mfdataset('../specific_humidity/*.nc', combine='by_coords')
t = xr.open_mfdataset('../temperature/*.nc', combine='by_coords')
tp = xr.open_mfdataset('../total_precipitation/*.nc', combine='by_coords')
u = xr.open_mfdataset('../u_component_of_wind/*.nc', combine='by_coords')
vo = xr.open_mfdataset('../vorticity/*.nc', combine='by_coords')
v = xr.open_mfdataset('../v_component_of_wind/*.nc', combine='by_coords')

param = [z,pv,r,q,t,u,vo,v,tp]
data = xr.merge(param)


start_date = pd.Timestamp(year=2015, month=1, day=1)
end_date = pd.Timestamp(year=2016, month=12, day=31)


ds_train = data.sel(time=slice('2010', '2019'))

dic = OrderedDict({'z': None, 'pv': None, 'r' : None,'q': None,'t': None,'u': None,'vo': None,'v': None,'tp': None})
bs = 24
lead_time = 1
dg_train = DataGenerator(
    ds_train.sel(time=slice(start_date, end_date)), dic, lead_time, batch_size=bs, load=True)
print('data loaded')

# 定义模型
shape1 = [24, 1, 65, 32, 64]
numlayers = 4 # numlayers 取值为 4
model = RNN(shape1, numlayers, [65, 65, 65, 65], 6, True)

model.load_state_dict(torch.load('model.pth'))

if torch.cuda.is_available():
    model.cuda()

# 定义损失函数和优化器
error = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1
for epoch in range(epochs):
  for i in range(len(dg_train)):
      x, y = dg_train[i]
      # B, C, T, H, W
      x_tensor = x.permute(0, 3, 1, 2)
      x_tensor = x_tensor.unsqueeze(1).to(device)
      # B, T, C, H, W
      # Clear gradients
      optimizer.zero_grad()
      # Forward propagation
      outputs = model(x_tensor).to(device) # B, C, T, H, W
      y = y.permute(0, 3, 1, 2)
      y = y.unsqueeze(1).to(device)  # B, T, C, H, W
      # Calculate loss
      loss = error(outputs, y)
      # Calculating gradients
      loss.backward()
      # Update parameters
      optimizer.step()
      print(outputs,y)
      print(loss)
  print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
torch.save(model.state_dict(), 'model.pth')
      
