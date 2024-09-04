import pandas as pd
import numpy as np
from chinese_calendar import is_workday  # 工作日判断
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
from datetime import datetime




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
upper_bound = Q3 + 1.5 * IQR

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
                                         'description_early', 'description_early1', 'time', 'date', 'DateTime'])
def remove_dates_from_texts(texts):
    # 删除日期
    date_pattern = r'\b\d{4}(?:年|\s)?(?:0?[1-9]|1[0-2])(?:月|\s)?(?:0?[1-9]|[12][0-9]|3[01])(?:日|\b)|\b(?:0?[1-9]|1[0-2])(?:月|\s)(?:0?[1-9]|[12][0-9]|3[01])日\b'
    return re.sub(date_pattern, '', texts).strip()

def remove_1_from_texts(texts):
    # 匹配 '- 10:20', '-10:20', ' 10:20', '10:20' 等形式
    time_split_pattern = r'(-\s*(\d{1,2}:\d{2}|\d{1,2}\.\d{2}))'  # 匹配时间范围的后半部分  # 匹配时间范围的后半部分及之后的任何内容
    return re.sub(time_split_pattern, '', texts).strip()

def process_text_with_times(text):
    # 使用正则表达式提取所有的时间信息
    pattern = r'(\d+:\d+)'
    matches = re.findall(pattern, text)

    if not matches:
        return text

    try:
        # 转换时间
        times = [datetime.strptime(match, '%H:%M') for match in matches]
        base_time = times[0]

        # 计算与第一个时间的分钟差，同时处理跨天情况
        time_differences = []
        for time in times:
            raw_diff = (time - base_time).total_seconds() // 60
            # 处理跨天情况
            if raw_diff < 0:
                # 计算从24:00到base_time，再从24:00到当前时间的总分钟数
                minutes_to_midnight = (24 * 60) - (base_time.hour * 60 + base_time.minute)
                # 从00:00到time的分钟数
                minutes_from_midnight = time.hour * 60 + time.minute
                diff = minutes_to_midnight + minutes_from_midnight
            else:
                diff = raw_diff
            time_differences.append(diff)

        # 构建替换逻辑，第一个时间映射为'0min'，其余时间为与第一个时间的分钟差
        replacements = {match: (f"{diff}min" if i > 0 else "0min") for i, (match, diff) in
                        enumerate(zip(matches, time_differences))}

    except ValueError as e:
        print(f"Error processing time: {e}")
        return text

    # 这里省略了文本替换逻辑的具体实现，因为直接在原始代码上修改并添加跨天处理是主要目的
    for match, replacement in replacements.items():
        text = re.sub(re.escape(match), replacement, text, count=1)

    return text


accident_data['description'] = accident_data['description'].apply(remove_dates_from_texts)
accident_data['description'] = accident_data['description'].apply(remove_1_from_texts)
#accident_data['description'] = accident_data['description'].apply(process_text_with_times)


categorical_columns = ['Weekday', 'Infrastructure_damage', 'Injury', 'Death', 'Vehicle_type', 'Vehicle_involved',
                      'Pavement_condition', 'Weather_condition', 'Shoulder', 'Burning', 'Rollover', 'Night_hours',
                      'Peak_hours', 'Ramp']
#stats_summary = pd.DataFrame(columns=['Column', 'Count_0', 'Count_1', 'Mean_0', 'Std_0', 'Mean_1', 'Std_1'])

duration = accident_data.pop('duration')
#duration.to_csv("duration_data0.01.csv", index=False)
# 划分训练集、验证集与测试集
train_val_data, test_data, train_val_duration, test_duration = train_test_split(accident_data, duration, test_size=0.15, random_state=42, shuffle=True)
train_data, val_data, train_duration, val_duration = train_test_split(train_val_data, train_val_duration, test_size=0.15, random_state=42, shuffle=True)

#对标签进行对数变换，使其近似正态分布

train_duration_log = np.log(train_duration.values)
val_duration_log = np.log(val_duration.values)
test_duration_log = np.log(test_duration.values)
train_duration = train_duration_log
val_duration = val_duration_log
test_duration = test_duration_log


# scaler = MinMaxScaler()
# train_duration_norm = scaler.fit_transform(train_duration.values.reshape(-1, 1))
# val_duration_norm = scaler.transform(val_duration.values.reshape(-1, 1))
# test_duration_norm = scaler.transform(test_duration.values.reshape(-1, 1))
# train_duration = train_duration_norm.squeeze()
# val_duration = val_duration_norm.squeeze()
# test_duration = test_duration_norm.squeeze()


# 提取文本数据做单独处理
# train_data_text = train_data["description_early1"]
# val_data_text = val_data["description_early1"]
# test_data_text = test_data["description_early1"]
train_data_text = train_data.pop("description")
val_data_text = val_data.pop("description")
test_data_text = test_data.pop("description")
# # 分类变量onehot编码
# train_data_onehot = pd.get_dummies(train_data, columns=categorical_columns)  # shape: (n_samples, n_features),
# val_data_onehot = pd.get_dummies(val_data, columns=categorical_columns)
# test_data_onehot = pd.get_dummies(test_data, columns=categorical_columns)


# 构建数据集类
class AccidentsDataset(Dataset):
    def __init__(self, accident_descriptions, cat_data, durations, tokenizer, max_length=256):
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
assert len(train_data_text) == len(train_data) == len(train_duration), "The lengths of input lists do not match."
# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('miniRBT')

# 构造数据集
train_dataset = AccidentsDataset(train_data_text, train_data, train_duration, tokenizer, max_length=256)
val_dataset = AccidentsDataset(val_data_text, val_data, val_duration, tokenizer, max_length=256)
test_dataset = AccidentsDataset(test_data_text, test_data, test_duration, tokenizer, max_length=256)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# 定义模型
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
            #nn.Linear(64, 64),
            #nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, out_features),
        )


    def forward(self, input_ids, attention_mask, categorical_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #pooled_sequence = outputs.last_hidden_state.mean(dim=1)
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        #pooled_sequence = self.dropout(pooled_sequence)
        text_features = torch.relu(self.fc1(cls_token_output))
        combined_features = torch.cat((text_features, categorical_features), dim=1)
        out = self.regression_layer(combined_features)
        return out


# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertDurationRegressor(train_data.shape[1])
model.to(device)

# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# lr = 0.0002
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
# #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
# epochs = 20
# patience = 5  # 早停epoch设置
# no_improvement_count = 0
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
#
# def train_epoch(model, data_loader, optimizer, criterion):
#     model.train()
#     total_loss = 0.0
#     for batch in data_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         categorical_features = batch['cat_data'].to(device)
#         targets = batch['target_duration'].to(device)
#         outputs = model(input_ids, attention_mask, categorical_features)
#         loss = criterion(outputs, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * input_ids.size(0)
#     return total_loss / len(data_loader.dataset)
# def val_epoch(model, data_loader, criterion):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for batch in data_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             categorical_features = batch['cat_data'].to(device)
#             targets = batch['target_duration'].to(device)  #
#             outputs = model(input_ids, attention_mask, categorical_features)
#             loss = criterion(outputs, targets)
#             total_loss += loss.item() * input_ids.size(0)
#     return total_loss / len(data_loader.dataset)
# best_val_loss = float('inf')
# train_losses = []
# val_losses = []
#
# columns = ['Epoch', 'Train_Loss', 'Val_Loss']
# training_log = pd.DataFrame(columns=columns)
#
# for epoch in range(epochs):
#     train_loss = train_epoch(model, train_loader, optimizer, criterion)
#     val_loss = val_epoch(model, val_loader, criterion)
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
#     scheduler.step()
#     torch.save(model.state_dict(), 'MiniRBT_fullmsg_binary.pth')
#     new_row = {'Epoch': epoch + 1, 'Train_Loss': train_loss, 'Val_Loss': val_loss}
#     #training_log = training_log.append(new_row, ignore_index=True)
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), 'MiniRBT_fullmsg_binary.pth')
#         no_improvement_count = 0
#     else:
#         no_improvement_count += 1
#         if no_improvement_count >= patience:
#             print(f'Early stopping triggered at epoch {epoch}. No improvement in validation loss for {patience} epochs.')
#             break
#     print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
# training_log.to_csv('training_log_lognormal.csv', index=False)
# print("Training log has been saved to 'training_log.csv'")
# 绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

#加载最佳模型权重
print('----------loading model---------------')
model.load_state_dict(torch.load('MiniRBT_fullmsg_binary.pth'))
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
predictions_original_scale = np.exp(test_preds)
actuals_original_scale = np.exp(test_labels)
# predictions_original_scale = scaler.inverse_transform(test_preds.reshape(-1, 1))
# actuals_original_scale = scaler.inverse_transform(test_labels.reshape(-1, 1))

# 计算指标
rmse, mae, mape = calculate_metrics(predictions_original_scale, actuals_original_scale)
print(f"Test Set Metrics: RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}%")
#
# df = pd.DataFrame({
#     'Predictions': predictions_original_scale.flatten(),  # 展平数组作为列
#     'Actuals': actuals_original_scale.flatten()
# })
#
# # 指定保存的文件名和路径
# csv_file_path = 'predictions_vs_actuals_initial.csv'
#
# # 将 DataFrame 写入 CSV 文件
# df.to_csv(csv_file_path, index=False)  # index=False 可以避免保存索引列
#
time_intervals = [(0,30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 185)]
interval_results = []
for interval in time_intervals:
    interval_mask = (actuals_original_scale >= interval[0]) & (actuals_original_scale <= interval[1])
    interval_preds = predictions_original_scale[interval_mask]
    interval_actuals = actuals_original_scale[interval_mask]

    if not interval_preds.size or not interval_actuals.size:
        print(f"No data in the interval {interval[0]}-{interval[1]} minutes.")
        continue

    rmse, mae, mape = calculate_metrics(interval_preds, interval_actuals)
    interval_results.append({
        "Interval": f"{interval[0]}-{interval[1]} min",
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    })

# 创建DataFrame并保存到CSV文件
df = pd.DataFrame(interval_results)
df.to_csv("group_metrics/MiniRBT+MLP_group_metrics.csv", index=False)