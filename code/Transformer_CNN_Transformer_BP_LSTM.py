import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 设置随机种子以便结果可重复
torch.manual_seed(0)

# 模拟时间序列数据
seq_length = 1680
input_size = 3
output_size = 1

# 生成示例数据
# Generate example data
df=pd.read_csv('Daily_RMSE.csv')
column_names=['Doy','GPT2w_ZTD','Daily_RMSE']
Input=df[column_names]
Output=df['GAMIT_ZTD']
data=np.array(Input)
target=np.array(Output)

# 将数据转换为PyTorch张量
data_tensor = torch.FloatTensor(data)
target_tensor = torch.FloatTensor(target)

# 将数据移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = data_tensor.to(device)
target_tensor = target_tensor.to(device)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出
        return out

# 初始化模型
hidden_size = 64
num_layers = 2
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

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
for epoch in range(num_epochs):
    mae_sum = 0.0
    mse_sum = 0.0
    bias_sum = 0.0

    for i in range(seq_length):
        inputs = data_tensor[i:i+1].unsqueeze(0)  # 添加unsqueeze(0)以匹配模型的输入维度
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

# 使用模型进行预测并计算预测结果的 MAE 和 MSE
predicted_outputs = []
predicted_mae_sum = 0.0
predicted_mse_sum = 0.0
predicted_bias_sum = 0.0
predicted_mae_concate_LSTM=[]
predicted_mse_concate_LSTM=[]
with torch.no_grad():
    for i in range(seq_length):
        test_input = data_tensor[i:i+1].unsqueeze(0)  # 添加unsqueeze(0)以匹配模型的输入维度
        predicted_output = model(test_input)

        # 计算预测结果的 MAE 和 MSE
        predicted_mae = torch.abs(predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_mse = ((predicted_output - target_tensor[i:i+1]) ** 2).mean().item()
        predicted_mae_concate_LSTM.append(predicted_mae)
        predicted_mse_concate_LSTM.append(predicted_mse)
        predicted_mae_sum += predicted_mae
        predicted_mse_sum += predicted_mse

        # 计算预测结果的偏差（bias）
        predicted_bias = (predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_bias_sum += predicted_bias

    # 计算平均预测 MAE 和 MSE
    average_predicted_mae = predicted_mae_sum / seq_length
    average_predicted_mse = predicted_mse_sum / seq_length
    average_predicted_bias = predicted_bias_sum / seq_length

# 绘制每时刻的 MAE 和 MSE 曲线
mae_values_LSTM=mae_values
mse_values_LSTM=mse_values
plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(mae_values_LSTM, label='MAE', color='blue')
plt.xlabel('Time Step')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE vs Time Step')

plt.subplot(4, 1, 2)
plt.plot(mse_values_LSTM, label='MSE', color='red')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE vs Time Step')

plt.subplot(4, 1, 3)
plt.plot(predicted_mae_concate_LSTM, label='MSE', color='red')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE vs Time Step')

plt.subplot(4, 1, 4)
plt.plot(predicted_mse_concate_LSTM, label='MSE', color='red')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE vs Time Step')
plt.tight_layout()
plt.show()
plt.savefig('All four in LSTM.png')

# 打印和绘制损失函数的曲线
plt.figure(figsize=(8, 4))
plt.plot(range(num_epochs), loss_values, label='Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epoch')
plt.show()
plt.savefig('Loss vs Epoch_LSTM.png')

# 打印训练结果和预测结果的偏差（bias）
print(f'Training Bias in LSTM is: {average_bias:.4f}')
print(f'Predicted Bias in LSTM is: {average_predicted_bias:.4f}')
print(f'average_predicted_mae in LSTM is: {average_predicted_mae:.4f}')
print(f'average_predicted_mse in LSTM is: {average_predicted_mse:.4f}')




import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便结果可重复
torch.manual_seed(0)

# 模拟时间序列数据
seq_length = 1680
input_size = 3
output_size = 1

# 生成示例数据
# 生成示例数据
df=pd.read_csv('Daily_RMSE.csv')
column_names=['Doy','GPT2w_ZTD','Daily_RMSE']
Input=df[column_names]
Output=df['GAMIT_ZTD']
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
mse_values_AR=[]
mae_values_AR=[]
mae_values_AR=mae_values
mse_values_AR=mse_values
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
predicted_mae_concate_AR= predicted_mae_concate
predicted_mse_concate_AR= predicted_mse_concate
# 计算平均预测 MAE 和 MSE
average_predicted_mae = predicted_mae_sum / seq_length
average_predicted_mse = predicted_mse_sum / seq_length
average_predicted_bias = predicted_bias_sum / seq_length
print("The average predict mae, mse and bias of AutoRegressiveTransformer are:",average_predicted_mae,average_predicted_mse,average_predicted_bias)





###Transformer
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch

# Set a random seed for reproducibility
torch.manual_seed(0)

# Simulate time series data
seq_length = 1680
input_size = 3
output_size = 1

# Generate example data
df=pd.read_csv('Daily_RMSE.csv')
column_names=['Doy','GPT2w_ZTD','Daily_RMSE']
Input=df[column_names]
Output=df['GAMIT_ZTD']
data=np.array(Input)
target=np.array(Output)

# Convert data to PyTorch tensors
data_tensor = torch.FloatTensor(data)
target_tensor = torch.FloatTensor(target)

# Move data and model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = data_tensor.to(device)
target_tensor = target_tensor.to(device)

# Define a Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, output_size):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead=2, num_encoder_layers=2)
        self.fc = nn.Linear(input_size, d_model)
        self.output = nn.Linear(d_model, output_size)

    def forward(self, src, tgt):
        src = self.fc(src)
        tgt = self.fc(tgt)
        out = self.transformer(src, tgt)
        out = self.output(out)
        return out

# Initialize the model and move it to the GPU
d_model = 64  # Transformer's dimension
model = TransformerModel(input_size, d_model, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store metrics
mae_values = []
mse_values = []
bias_values = []
loss_values = []

# Training the model
num_epochs = 30
for epoch in range(num_epochs):
    mae_sum = 0.0
    mse_sum = 0.0
    bias_sum = 0.0

    for i in range(seq_length):
        inputs = data_tensor[i:i+1]
        targets = target_tensor[i:i+1]

        optimizer.zero_grad()
        outputs = model(inputs, inputs)  # Use the same input for both src and tgt
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Calculate MAE and MSE
        mae = torch.abs(outputs - targets).mean().item()
        mse = ((outputs - targets) ** 2).mean().item()

        mae_sum += mae
        mse_sum += mse

        # Calculate bias
        bias = (outputs - targets).mean().item()
        bias_sum += bias

        # Store MAE and MSE at each time step
        mae_values.append(mae)
        mse_values.append(mse)

    # Calculate average MAE and MSE
    average_mae = mae_sum / seq_length
    average_mse = mse_sum / seq_length
    average_bias = bias_sum / seq_length

    loss_values.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average MAE: {average_mae:.4f}, Average MSE: {average_mse:.4f}, Average Bias: {average_bias:.4f}')

# Use the model for prediction and calculate MAE, MSE, and bias
predicted_outputs = []
predicted_mae_sum = 0.0
predicted_mse_sum = 0.0
predicted_bias_sum = 0.0
predicted_mae_concate=[]
predicted_mse_concate=[]
with torch.no_grad():
    for i in range(seq_length):
        test_input = data_tensor[i:i+1]
        predicted_output = model(test_input, test_input)

        # Calculate MAE, MSE, and bias for the predicted outputs
        predicted_mae = torch.abs(predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_mse = ((predicted_output - target_tensor[i:i+1]) ** 2).mean().item()

        predicted_mae_sum += predicted_mae
        predicted_mse_sum += predicted_mse
        predicted_mae_concate.append(predicted_mae)
        predicted_mse_concate.append(predicted_mse)
        predicted_bias = (predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_bias_sum += predicted_bias

# Calculate average MAE, MSE, and bias for the predicted outputs
average_predicted_mae = predicted_mae_sum / seq_length
average_predicted_mse = predicted_mse_sum / seq_length
average_predicted_bias = predicted_bias_sum / seq_length

# Plot MAE and MSE at each time step
mae_values_Transformer=[]
mse_values_Transformer=[]
predicted_mae_concate_Transformer=[]
predicted_mse_concate_Transformer=[]
mae_values_Transformer=mae_values
mse_values_Transformer=mse_values
predicted_mae_concate_Transformer=predicted_mae_concate
predicted_mse_concate_Transformer=predicted_mse_concate
plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(mae_values_Transformer, label='MAE', color='blue')
plt.xlabel('Time Step')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE vs Time Step')

plt.subplot(4, 1, 2)
plt.plot(mse_values_Transformer, label='MSE', color='red')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE vs Time Step')

plt.subplot(4, 1, 3)
plt.plot(predicted_mae_concate_Transformer, label='MAE', color='blue')
plt.xlabel('Time Step')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE vs Time Step')

plt.subplot(4, 1, 4)
plt.plot(predicted_mse_concate_Transformer, label='MSE', color='red')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE vs Time Step')

plt.tight_layout()
plt.show()

# 打印和绘制损失函数的曲线
plt.figure(figsize=(8, 4))
plt.plot(loss_values, label='Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epoch')
plt.show()

# 打印训练结果和预测结果的偏差（bias）
print(f'Training Bias of Transformer is : {average_bias:.4f}')
print("average_predicted_mae, average_predicted_mse, Predicted Bias of Transformer is :",average_predicted_mae,average_predicted_mse,average_predicted_bias)








import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 创建一个示例的多变量时间序列数据集，长度为 1680，输入特征数为 3，输出为 1
# 这里使用随机数据作为示例
seq_length = 1680
input_size = 3
output_size = 1

# 生成示例数据
import pandas as pd
df=pd.read_csv('Daily_RMSE.csv')
df.head()
column_names=['Doy','GPT2w_ZTD','Daily_RMSE']
Input=df[column_names]
Output=df['GAMIT_ZTD']
data=np.array(Input)
target=np.array(Output)
# data = np.random.randn(seq_length, input_size)
# target = np.random.randn(seq_length, output_size)

# 将数据转换为 PyTorch 张量，并移动到GPU
data_tensor = torch.FloatTensor(data).to('cuda')
target_tensor = torch.FloatTensor(target).to('cuda')

# 定义一个简单的多层感知器 (MLP) 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 初始化模型并将其移动到GPU
hidden_size = 64  # 隐藏层神经元数量
model = MLP(input_size, hidden_size, output_size).to('cuda')

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 存储每时刻的 MAE 和 MSE
train_mae_values = []
train_mse_values = []

# 存储每时刻的预测 MAE 和 MSE
predicted_mae_values = []
predicted_mse_values = []

# 存储每时刻的偏差（bias）
train_bias_values = []
predicted_bias_values = []

# 存储损失函数值
loss_values = []

# 训练模型
num_epochs = 30
for epoch in range(num_epochs):
    mae_sum = 0.0
    mse_sum = 0.0
    bias_sum = 0.0

    # 前向传播和反向传播
    for i in range(seq_length):
        inputs = data_tensor[i:i+1]
        targets = target_tensor[i:i+1]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 计算训练结果的 MAE 和 MSE
        mae = torch.abs(outputs - targets).mean().item()
        mse = ((outputs - targets) ** 2).mean().item()

        mae_sum += mae
        mse_sum += mse

        # 计算训练结果的偏差（bias）
        bias = (outputs - targets).mean().item()
        bias_sum += bias

        # 存储每时刻的 MAE 和 MSE
        train_mae_values.append(mae)
        train_mse_values.append(mse)
        train_bias_values.append(bias)

    # 计算平均 MAE 和 MSE
    average_mae = mae_sum / seq_length
    average_mse = mse_sum / seq_length
    average_bias = bias_sum / seq_length

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average MAE: {average_mae:.4f}, Average MSE: {average_mse:.4f}, Average Bias: {average_bias:.4f}')

    loss_values.append(loss.item())

# 使用模型进行预测并计算预测结果的 MAE 和 MSE
predicted_outputs = []
predicted_mae_sum = 0.0
predicted_mse_sum = 0.0
predicted_bias_sum = 0.0

with torch.no_grad():
    for i in range(seq_length):
        test_input = data_tensor[i:i+1]
        predicted_output = model(test_input)

        # 计算预测结果的 MAE 和 MSE
        predicted_mae = torch.abs(predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_mse = ((predicted_output - target_tensor[i:i+1]) ** 2).mean().item()

        predicted_mae_sum += predicted_mae
        predicted_mse_sum += predicted_mse

        # 计算预测结果的偏差（bias）
        predicted_bias = (predicted_output - target_tensor[i:i+1]).mean().item()
        predicted_bias_sum += predicted_bias

        # 存储每时刻的预测 MAE 和 MSE
        predicted_mae_values.append(predicted_mae)
        predicted_mse_values.append(predicted_mse)
        predicted_bias_values.append(predicted_bias)

    # 计算平均预测 MAE 和 MSE
    average_predicted_mae = predicted_mae_sum / seq_length
    average_predicted_mse = predicted_mse_sum / seq_length
    average_predicted_bias = predicted_bias_sum / seq_length

# 绘制每时刻的 MAE 和 MSE 曲线
mae_values_BP=[]
mse_values_BP=[]
predicted_mae_values_BP=[]
predicted_mse_values_BP=[]
mae_values_BP=train_mae_values
mse_values_BP=train_mse_values
predicted_mae_values_BP=predicted_mae_values
predicted_mse_values_BP=predicted_mse_values
plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(mae_values_BP, label='Train MAE', color='blue')
plt.xlabel('Time Step')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE vs Time Step')

plt.subplot(4, 1, 2)
plt.plot(mse_values_BP, label='Train MSE', color='blue')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE vs Time Step')

plt.subplot(4, 1, 3)
plt.plot(predicted_mae_values_BP, label='Train MAE', color='blue')
plt.xlabel('Time Step')
plt.ylabel('MAE')
plt.legend()
plt.title('MAE vs Time Step')

plt.subplot(4, 1, 4)
plt.plot(predicted_mse_values_BP, label='Train MSE', color='blue')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE vs Time Step')
plt.tight_layout()
plt.show()

# 打印和绘制损失函数的曲线
plt.figure(figsize=(8, 4))
plt.plot(range(num_epochs), loss_values, label='Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epoch')
plt.show()

# 打印训练结果和预测结果的偏差（bias）
print(f'Training Bias of BP is : {average_bias:.4f}, Predicted Bias of BP is: {average_predicted_bias:.4f}')
print(f'average_predicted_mae of BP is : {average_predicted_mae:.4f}, average_predicted_mse of BP is: {average_predicted_mse:.4f}')


import csv

# Create or open a CSV file for writing
with open('results.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Example variables

    # Write the variables to a CSV file
    csvwriter.writerow([mae_values_BP,mse_values_BP
,predicted_mae_values_BP
,predicted_mse_values_BP,
mae_values_Transformer
,mse_values_Transformer,predicted_mae_concate_Transformer,predicted_mse_concate_Transformer,mae_values_LSTM,mse_values_LSTM
,predicted_mae_concate_LSTM
,predicted_mse_concate_LSTM,mae_values_AR
,mse_values_AR,predicted_mae_concate_AR
,predicted_mse_concate_AR])

    # You can write more rows of data as needed

# Close the CSV file
csvfile.close()