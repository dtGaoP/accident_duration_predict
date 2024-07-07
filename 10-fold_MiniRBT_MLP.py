import pandas as pd
import numpy as np
from chinese_calendar import is_workday  # 工作日判断
import matplotlib.pyplot as plt  # 绘图库
import jieba  # 中文分词
import re  # 正则化
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.preprocessing import MinMaxScaler  # 归一化处理
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学习率调整策略

from sklearn.metrics import mean_squared_error, mean_absolute_error  # 评价指标
from torch.utils.data import Dataset
from sklearn.model_selection import KFold




pd.set_option('display.max_columns', None)  # 显示全部列

df = pd.read_csv('accident_data_new1.csv',encoding='gbk')
# 删除多列缺失值
columns_to_check = ['location_type', 'weather', 'environment_condition', 'vehicle', 'impact', 'death_num', 'injury_num', 'duration_h', 'description']
df = df.dropna(subset=columns_to_check, how='any', axis=0)
#删除重复值
df = df.drop_duplicates(keep='first')
# duration conversion
df['duration_h'] = pd.to_numeric(df['duration_h'], errors='coerce')
df['duration_min'] = pd.to_numeric(df['duration_min'], errors='coerce')
df = df.dropna(subset=['duration_h', 'duration_min'])
df['duration'] = df['duration_h'] * 60 + df['duration_min']
# duration outliers delete
Q1 = df['duration'].quantile(0.25)  # 第一四分位数
Q3 = df['duration'].quantile(0.75)  # 第三四分位数
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 0.5 * IQR

df_cleaned = df[(df['duration'] >= lower_bound) & (df['duration'] <= upper_bound)]
print("原始数据长度:", len(df))
print("去除异常值后的数据长度:", len(df_cleaned))

# time conversion
df_cleaned['date'] = pd.to_datetime(df_cleaned[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
df_cleaned['time'] = df_cleaned['start_time'] + ':00'
df_cleaned['time'] = pd.to_timedelta(df_cleaned['time'])
df_cleaned['DateTime'] = df_cleaned['date'] + df_cleaned['time']
# weekday
df_cleaned['Weekday'] = df_cleaned['DateTime'].apply(is_workday)
df_cleaned['Weekday'] = df_cleaned['Weekday'].astype(int)
# infrastructure damage
df_cleaned['Infrastructure_damage'] = df_cleaned['description'].str.contains('有路产', case=False).astype(int)
# injury
df_cleaned['Injury'] = (df_cleaned['injury_num'] > 0).astype(int)
# death
df_cleaned['Death'] = (df_cleaned['death_num'] > 0).astype(int)
# vehicle_type
df_cleaned['Vehicle_type'] = (
    df_cleaned['vehicle'].str.contains("一型客车", case=False) &
    ~df_cleaned['vehicle'].str.contains('|'.join(["货车", "半挂", "皮卡"]), case=False)
).astype(int)
# vehicle_involved
def count_one_vehicle(text):
    count_one = text.count('一辆')
    has_and = '与' in text
    return 0 if count_one == 1 and not has_and else 1


df_cleaned['Vehicle_involved'] = df_cleaned['vehicle'].apply(count_one_vehicle)
# Pavement_condition
pavement_normal_conditions = ['A', 'D', 'E', 'F']
df_cleaned['Pavement_condition'] = np.where(df_cleaned['environment_condition'].isin(pavement_normal_conditions), 0, 1)
# Weather_condition
df_cleaned['Weather_condition'] = np.where(df_cleaned['weather'].isin(['晴', '阴']), 0, 1)
# Shoulder
df_cleaned['Shoulder'] = (
    df_cleaned['impact'].str.contains('|'.join(["应急车道", "不影响", "不占用", "收费站", "服务区"]), case=False) &
    ~df_cleaned['impact'].str.contains('|'.join(["和", "与", "行车道", "超车道", "第一", "第二", "1", "2", "3", "4"]), case=False)
).astype(int)
# Burning
df_cleaned['Burning'] = df_cleaned['description'].str.contains('|'.join(['自燃', '燃烧', '火情', '起火']), case=False).astype(int)
# Rollover
df_cleaned['Rollover'] = df_cleaned['description'].str.contains('侧翻', case=False).astype(int)
# Night_hours
df_cleaned['DateTime'] = pd.to_datetime(df_cleaned['DateTime'])
df_cleaned['Night_hours'] = ((df_cleaned['DateTime'].dt.hour >= 20) | (df_cleaned['DateTime'].dt.hour < 6)).astype(int)
# Peak_hours
df_cleaned['Peak_hours'] = ((df_cleaned['DateTime'].dt.hour >= 6) & (df_cleaned['DateTime'].dt.hour < 9) |
                    (df_cleaned['DateTime'].dt.hour >= 17) & (df_cleaned['DateTime'].dt.hour < 20)).astype(int)
# Ramp
df_cleaned['Ramp'] = (df_cleaned['location_type'].str.contains('D')).astype(int)
# drop unrelated columns
accident_data = df_cleaned.drop(columns=['year', 'month', 'day', 'start_time', 'location_type', 'weather', 'direction', 'environment_condition',
                                         'event_type', 'vehicle', 'accident_type', 'impact_location', 'impact',
                                         'death_num', 'injury_num', 'end_time', 'duration_h', 'duration_min',
                                         'description_early', 'description', 'time', 'date', 'DateTime'])

categorical_columns = ['Weekday', 'Infrastructure_damage', 'Injury', 'Death', 'Vehicle_type', 'Vehicle_involved',
                      'Pavement_condition', 'Weather_condition', 'Shoulder', 'Burning', 'Rollover', 'Night_hours',
                      'Peak_hours', 'Ramp']

duration = accident_data.pop('duration')

train_val_data, test_data, train_val_duration, test_duration = train_test_split(accident_data, duration, test_size=0.20, random_state=42, shuffle=True)


train_val_text = train_val_data.pop("description_early1")
test_text = test_data.pop("description_early1")

# 定义编码器
tokenizer = BertTokenizer.from_pretrained('miniRBT')
batch_size = 16


class AccidentsDataset(Dataset):
    def __init__(self, accident_descriptions, cat_data, durations, tokenizer, max_length=128):
        self.accident_descriptions = list(accident_descriptions)
        self.cat_data = cat_data
        self.durations = list(durations)  # Convert to lists if they were pandas Series or DataFrames with indices
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.accident_descriptions)


    def __getitem__(self, index):
        #print(index)
        accident_descriptions = self.accident_descriptions[index]
        cat_data = self.cat_data.iloc[index].values
        duration = self.durations[index]
        inputs = self.tokenizer(accident_descriptions, padding='max_length',
                                truncation=True, max_length=self.max_length, return_tensors='pt')
        cat_data = torch.tensor(cat_data)
        target_duration = torch.tensor([duration], dtype=torch.float)

        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'cat_data': cat_data,
            'target_duration': target_duration
        }


class BertDurationRegressor(nn.Module):
    def __init__(self, categorical_features_size, out_features=1):
        super().__init__()
        self.bert_hidden_dim = bert_hidden_dim = 256
        self.dense_size = dense_size = 128
        self.dropout_rate = dropout_rate = 0.1
        self.dropout = nn.Dropout(dropout_rate)
        self.bert = BertModel.from_pretrained('miniRBT')
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc1 = nn.Linear(in_features=bert_hidden_dim,
                             out_features=dense_size
                             )
        self.regression_layer = nn.Sequential(
            nn.Linear(dense_size + categorical_features_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, out_features),
        )


    def forward(self, input_ids, attention_mask, categorical_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_sequence = outputs.last_hidden_state.mean(dim=1)
        pooled_sequence = self.dropout(pooled_sequence)
        text_features = torch.relu(self.fc1(pooled_sequence))
        combined_features = torch.cat((text_features, categorical_features), dim=1)
        out = self.regression_layer(combined_features)
        return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertDurationRegressor(train_val_data.shape[1]).to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
lr = 0.0002
epochs = 20
patience = 5  # 早停epoch设置
no_improvement_count = 0

def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        categorical_features = batch['cat_data'].to(device)
        targets = batch['target_duration'].to(device)
        outputs = model(input_ids, attention_mask, categorical_features)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(data_loader.dataset)
def val_epoch(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            categorical_features = batch['cat_data'].to(device)
            targets = batch['target_duration'].to(device)  #
            outputs = model(input_ids, attention_mask, categorical_features)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(data_loader.dataset)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_val_losses = []
best_val_loss = float('inf')
best_model_weights = None

# 初始化列表来保存每个fold的最佳验证损失
fold_best_val_losses = []

for fold, (train_indices, val_indices) in enumerate(kfold.split(train_val_data)):
    print(f"Fold {fold + 1}/{kfold.n_splits}")

    train_fold_data = train_val_data.iloc[train_indices]
    val_fold_data = train_val_data.iloc[val_indices]
    train_fold_duration = train_val_duration.iloc[train_indices]
    val_fold_duration = train_val_duration.iloc[val_indices]
    train_fold_text = train_val_text.iloc[train_indices]
    val_fold_text = train_val_text.iloc[val_indices]
# 标签处理（归一化or lg变换）
#     scaler = MinMaxScaler()
#     train_duration_norm = scaler.fit_transform(train_fold_duration.values.reshape(-1, 1))
#     val_duration_norm = scaler.transform(val_fold_duration.values.reshape(-1, 1))
#     train_fold_duration = train_duration_norm.squeeze()
#     val_fold_duration = val_duration_norm.squeeze()

    train_fold_dataset = AccidentsDataset(train_fold_text, train_fold_data, train_fold_duration, tokenizer, max_length=128)
    val_fold_dataset = AccidentsDataset(val_fold_text, val_fold_data, val_fold_duration, tokenizer, max_length=128)

    train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion)
        with torch.no_grad():
            val_loss = val_epoch(model, val_loader, criterion)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()  # 使用state_dict的copy方法

        # 收集验证损失
        cv_val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}")
# 将当前fold的最佳验证损失存入列表
    fold_best_val_losses.append(best_val_loss)
    # 重置最佳验证损失，以便下一个fold的计算
    best_val_loss = float('inf')


average_val_loss = np.mean(fold_best_val_losses)
print(f"Average Validation Loss across {kfold.n_splits} folds: {average_val_loss:.4f}")

# 使用平均最佳损失对应的模型权重恢复模型
model.load_state_dict(best_model_weights)
# 保存模型
torch.save(model.state_dict(), 'best_model_across_folds.pth')

print("Best model across all folds saved.")



#对标签进行对数变换，使其近似正态分布

# train_duration_log = np.log(train_duration.values)
# val_duration_log = np.log(val_duration.values)
# test_duration_log = np.log(test_duration.values)
# train_duration = train_duration_log
# val_duration = val_duration_log
# test_duration = test_duration_log

# 加载测试集

# scaler = MinMaxScaler()
# test_duration_norm = scaler.fit_transform(test_duration.values.reshape(-1, 1))
# test_duration = test_duration_norm.squeeze()
#
test_dataset = AccidentsDataset(test_text, test_data, test_duration, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 加载最佳模型权重
print('----------loading model---------------')
model.load_state_dict(torch.load('best_model_across_folds.pth'))
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        categorical_features = batch['cat_data'].to(device)
        targets = batch['target_duration'].to(device)
        outputs = model(inputs, attention_mask, categorical_features)
        test_preds.extend(outputs.cpu().detach().numpy())
        test_labels.extend(targets.cpu().detach().numpy())


def calculate_metrics(predictions, targets):
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100  # 注意处理除零的情况
    return rmse, mae, mape


# 转换为NumPy数组
# test_preds = np.array(outputs)
# test_labels = np.array(test_labels)
test_preds = np.concatenate(test_preds, axis=0)  # 将列表转换为单个NumPy数组
test_labels = np.concatenate(test_labels, axis=0)
# predictions_original_scale = np.exp(test_preds)
# actuals_original_scale = np.exp(test_labels)
# predictions_original_scale = scaler.inverse_transform(test_preds.reshape(-1, 1))
# actuals_original_scale = scaler.inverse_transform(test_labels.reshape(-1, 1))

# 计算指标
rmse, mae, mape = calculate_metrics(test_preds, test_labels)
print(f"Test Set Metrics: RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}%")