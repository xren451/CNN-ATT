
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# 设置随机种子以便结果可重复
torch.manual_seed(0)

# 模拟时间序列数据
seq_length = 17420
input_size = 6
output_size = 1

# 生成示例数据
# 生成示例数据
df=pd.read_csv('Daily_RMSE - ETTh1.csv')
column_names=['HUFL','HULL','MUFL','MULL','LUFL','LULL']
Input=df[column_names]
Output=df['OT']
data=np.array(Input)
target=np.array(Output)

# 将数据转换为PyTorch张量
data_tensor = torch.FloatTensor(data)
target_tensor = torch.FloatTensor(target)

# 将数据移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = data_tensor.to(device)
target_tensor = target_tensor.to(device)

# 定义AutoRegressive Transformer模型
class AutoRegressiveTransformer(nn.Module):
    def __init__(self, input_size, d_model, output_size, num_layers=2, nhead=2):
        super(AutoRegressiveTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(input_size, d_model)
        self.output = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.fc(x)
        out = self.transformer(x, x)  # AutoRegressive
        out = self.output(out)
        return out

# 初始化模型
d_model = 64  # Transformer的维度
num_layers = 2
nhead = 2
model = AutoRegressiveTransformer(input_size, d_model, output_size, num_layers, nhead).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 存储每时刻的 MAE 和 MSE
mae_values = []
mse_values = []

# 存储每时刻的偏差（bias）
bias_values = []

# 存储损失函数值
loss_values = []

# 训练模型
num_epochs = 30
training_start_time=time.time()
for epoch in range(num_epochs):
    mae_sum = 0.0
    mse_sum = 0.0
    bias_sum = 0.0

    for i in range(seq_length):
        inputs = data_tensor[i:i+1]
        targets = target_tensor[i:i+1]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 计算 MAE 和 MSE
        mae = torch.abs(outputs - targets).mean().item()
        mse = ((outputs - targets) ** 2).mean().item()

        mae_sum += mae
        mse_sum += mse

        # 计算偏差（bias）
        bias = (outputs - targets).mean().item()
        bias_sum += bias

        # 存储每时刻的 MAE 和 MSE
        mae_values.append(mae)
        mse_values.append(mse)

    # 计算平均 MAE 和 MSE
    average_mae = mae_sum / seq_length
    average_mse = mse_sum / seq_length
    average_bias = bias_sum / seq_length

    loss_values.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average MAE: {average_mae:.4f}, Average MSE: {average_mse:.4f}, Average Bias: {average_bias:.4f}')
training_end_time = time.time()  # Record end time for training
training_time = training_end_time - training_start_time
print(f'Training Time of ETTh1: {training_time:.2f} seconds')


mse_values_AR=[]
mae_values_AR=[]
mae_values_AR=mae_values
mse_values_AR=mse_values

# Inference
inference_start_time = time.time()  # Record start time for inference

# 使用模型进行预测并计算预测结果的 MAE 和 MSE
predicted_outputs = []
predicted_mae=[]
predicted_mse=[]
predicted_mae_sum = 0.0
predicted_mse_sum = 0.0
predicted_bias_sum = 0.0
predicted_mae_concate=[]
predicted_mse_concate=[]
with torch.no_grad():
    for i in range(seq_length):
        test_input = data_tensor[i:i+1]
        predicted_output = model(test_input)

        # 计算预测结果的 MAE 和 MSE
        predicted_mae = torch.abs(predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_mse = ((predicted_output - target_tensor[i:i+1]) ** 2).mean().item()

        predicted_mae_sum += predicted_mae
        predicted_mse_sum += predicted_mse
        predicted_mae_concate.append(predicted_mae)
        predicted_mse_concate.append(predicted_mse)
        # 计算预测结果的偏差（bias）
        predicted_bias = (predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_bias_sum += predicted_bias
inference_end_time = time.time()  # Record end time for inference
inference_time = inference_end_time - inference_start_time
print(f'Inference Time of ETTh1: {inference_time:.4f} seconds')
predicted_mae_concate_AR= predicted_mae_concate
predicted_mse_concate_AR= predicted_mse_concate
# 计算平均预测 MAE 和 MSE
average_predicted_mae = predicted_mae_sum / seq_length
average_predicted_mse = predicted_mse_sum / seq_length
average_predicted_bias = predicted_bias_sum / seq_length
print("The average predict mae, mse and bias of AutoRegressiveTransformer are:",average_predicted_mae,average_predicted_mse,average_predicted_bias)






import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# 设置随机种子以便结果可重复
torch.manual_seed(0)

# 模拟时间序列数据
seq_length = 17420
input_size = 6
output_size = 1

# 生成示例数据
# 生成示例数据
df=pd.read_csv('Daily_RMSE - ETTm2.csv')
column_names=['HUFL','HULL','MUFL','MULL','LUFL','LULL']
Input=df[column_names]
Output=df['OT']
data=np.array(Input)
target=np.array(Output)

# 将数据转换为PyTorch张量
data_tensor = torch.FloatTensor(data)
target_tensor = torch.FloatTensor(target)

# 将数据移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = data_tensor.to(device)
target_tensor = target_tensor.to(device)

# 定义AutoRegressive Transformer模型
class AutoRegressiveTransformer(nn.Module):
    def __init__(self, input_size, d_model, output_size, num_layers=2, nhead=2):
        super(AutoRegressiveTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(input_size, d_model)
        self.output = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.fc(x)
        out = self.transformer(x, x)  # AutoRegressive
        out = self.output(out)
        return out

# 初始化模型
d_model = 64  # Transformer的维度
num_layers = 2
nhead = 2
model = AutoRegressiveTransformer(input_size, d_model, output_size, num_layers, nhead).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 存储每时刻的 MAE 和 MSE
mae_values = []
mse_values = []

# 存储每时刻的偏差（bias）
bias_values = []

# 存储损失函数值
loss_values = []

# 训练模型
num_epochs = 30
training_start_time=time.time()
for epoch in range(num_epochs):
    mae_sum = 0.0
    mse_sum = 0.0
    bias_sum = 0.0

    for i in range(seq_length):
        inputs = data_tensor[i:i+1]
        targets = target_tensor[i:i+1]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 计算 MAE 和 MSE
        mae = torch.abs(outputs - targets).mean().item()
        mse = ((outputs - targets) ** 2).mean().item()

        mae_sum += mae
        mse_sum += mse

        # 计算偏差（bias）
        bias = (outputs - targets).mean().item()
        bias_sum += bias

        # 存储每时刻的 MAE 和 MSE
        mae_values.append(mae)
        mse_values.append(mse)

    # 计算平均 MAE 和 MSE
    average_mae = mae_sum / seq_length
    average_mse = mse_sum / seq_length
    average_bias = bias_sum / seq_length

    loss_values.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average MAE: {average_mae:.4f}, Average MSE: {average_mse:.4f}, Average Bias: {average_bias:.4f}')
training_end_time = time.time()  # Record end time for training
training_time = training_end_time - training_start_time
print(f'Training Time of ETTm2: {training_time:.2f} seconds')


mse_values_AR=[]
mae_values_AR=[]
mae_values_AR=mae_values
mse_values_AR=mse_values

# Inference
inference_start_time = time.time()  # Record start time for inference

# 使用模型进行预测并计算预测结果的 MAE 和 MSE
predicted_outputs = []
predicted_mae=[]
predicted_mse=[]
predicted_mae_sum = 0.0
predicted_mse_sum = 0.0
predicted_bias_sum = 0.0
predicted_mae_concate=[]
predicted_mse_concate=[]
with torch.no_grad():
    for i in range(seq_length):
        test_input = data_tensor[i:i+1]
        predicted_output = model(test_input)

        # 计算预测结果的 MAE 和 MSE
        predicted_mae = torch.abs(predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_mse = ((predicted_output - target_tensor[i:i+1]) ** 2).mean().item()

        predicted_mae_sum += predicted_mae
        predicted_mse_sum += predicted_mse
        predicted_mae_concate.append(predicted_mae)
        predicted_mse_concate.append(predicted_mse)
        # 计算预测结果的偏差（bias）
        predicted_bias = (predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_bias_sum += predicted_bias
inference_end_time = time.time()  # Record end time for inference
inference_time = inference_end_time - inference_start_time
print(f'Inference Time of ETTm2: {inference_time:.4f} seconds')
predicted_mae_concate_AR= predicted_mae_concate
predicted_mse_concate_AR= predicted_mse_concate
# 计算平均预测 MAE 和 MSE
average_predicted_mae = predicted_mae_sum / seq_length
average_predicted_mse = predicted_mse_sum / seq_length
average_predicted_bias = predicted_bias_sum / seq_length
print("The average predict mae, mse and bias of AutoRegressiveTransformer are:",average_predicted_mae,average_predicted_mse,average_predicted_bias)




import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# 设置随机种子以便结果可重复
torch.manual_seed(0)

# 模拟时间序列数据
seq_length = 184702
input_size = 4
output_size = 1

# 生成示例数据
# 生成示例数据
df=pd.read_csv('Daily_RMSE - ETTm2.csv')
column_names=['MUFL','MULL','LUFL','LULL']
Input=df[column_names]
Output=df['OT']
data=np.array(Input)
target=np.array(Output)

# 将数据转换为PyTorch张量
data_tensor = torch.FloatTensor(data)
target_tensor = torch.FloatTensor(target)

# 将数据移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = data_tensor.to(device)
target_tensor = target_tensor.to(device)

# 定义AutoRegressive Transformer模型
class AutoRegressiveTransformer(nn.Module):
    def __init__(self, input_size, d_model, output_size, num_layers=2, nhead=2):
        super(AutoRegressiveTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(input_size, d_model)
        self.output = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.fc(x)
        out = self.transformer(x, x)  # AutoRegressive
        out = self.output(out)
        return out

# 初始化模型
d_model = 64  # Transformer的维度
num_layers = 2
nhead = 2
model = AutoRegressiveTransformer(input_size, d_model, output_size, num_layers, nhead).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 存储每时刻的 MAE 和 MSE
mae_values = []
mse_values = []

# 存储每时刻的偏差（bias）
bias_values = []

# 存储损失函数值
loss_values = []

# 训练模型
num_epochs = 30
training_start_time=time.time()
for epoch in range(num_epochs):
    mae_sum = 0.0
    mse_sum = 0.0
    bias_sum = 0.0

    for i in range(seq_length):
        inputs = data_tensor[i:i+1]
        targets = target_tensor[i:i+1]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 计算 MAE 和 MSE
        mae = torch.abs(outputs - targets).mean().item()
        mse = ((outputs - targets) ** 2).mean().item()

        mae_sum += mae
        mse_sum += mse

        # 计算偏差（bias）
        bias = (outputs - targets).mean().item()
        bias_sum += bias

        # 存储每时刻的 MAE 和 MSE
        mae_values.append(mae)
        mse_values.append(mse)

    # 计算平均 MAE 和 MSE
    average_mae = mae_sum / seq_length
    average_mse = mse_sum / seq_length
    average_bias = bias_sum / seq_length

    loss_values.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average MAE: {average_mae:.4f}, Average MSE: {average_mse:.4f}, Average Bias: {average_bias:.4f}')
training_end_time = time.time()  # Record end time for training
training_time = training_end_time - training_start_time
print(f'Training Time of ETTm2: {training_time:.2f} seconds')


mse_values_AR=[]
mae_values_AR=[]
mae_values_AR=mae_values
mse_values_AR=mse_values

# Inference
inference_start_time = time.time()  # Record start time for inference

# 使用模型进行预测并计算预测结果的 MAE 和 MSE
predicted_outputs = []
predicted_mae=[]
predicted_mse=[]
predicted_mae_sum = 0.0
predicted_mse_sum = 0.0
predicted_bias_sum = 0.0
predicted_mae_concate=[]
predicted_mse_concate=[]
with torch.no_grad():
    for i in range(seq_length):
        test_input = data_tensor[i:i+1]
        predicted_output = model(test_input)

        # 计算预测结果的 MAE 和 MSE
        predicted_mae = torch.abs(predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_mse = ((predicted_output - target_tensor[i:i+1]) ** 2).mean().item()

        predicted_mae_sum += predicted_mae
        predicted_mse_sum += predicted_mse
        predicted_mae_concate.append(predicted_mae)
        predicted_mse_concate.append(predicted_mse)
        # 计算预测结果的偏差（bias）
        predicted_bias = (predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_bias_sum += predicted_bias
inference_end_time = time.time()  # Record end time for inference
inference_time = inference_end_time - inference_start_time
print(f'Inference Time of ETTm2: {inference_time:.4f} seconds')
predicted_mae_concate_AR= predicted_mae_concate
predicted_mse_concate_AR= predicted_mse_concate
# 计算平均预测 MAE 和 MSE
average_predicted_mae = predicted_mae_sum / seq_length
average_predicted_mse = predicted_mse_sum / seq_length
average_predicted_bias = predicted_bias_sum / seq_length
print("The average predict mae, mse and bias of AutoRegressiveTransformer are:",average_predicted_mae,average_predicted_mse,average_predicted_bias)
