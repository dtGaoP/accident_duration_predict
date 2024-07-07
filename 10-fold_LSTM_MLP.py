import pandas as pd
import numpy as np
from chinese_calendar import is_workday  # 工作日判断
import matplotlib.pyplot as plt  # 绘图库
import jieba  # 中文分词
import re  # 正则化
import os  # 读取文件
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.preprocessing import MinMaxScaler  # 归一化处理
from gensim.models import Word2Vec  # 词嵌入
from gensim.models.word2vec import LineSentence
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学习率调整策略
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 评价指标
import random




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

# description preprocess (text)
# 加载自定义词典
dict_folder = 'dict/'
# 遍历文件夹中的所有文件
for filename in os.listdir(dict_folder):
    if filename.endswith('.txt'):  # 确保只加载.txt文件
        dict_path = os.path.join(dict_folder, filename)
        jieba.load_userdict(dict_path)  # 加载每个词典文件

# 加载停用词列表
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())
# 去除车牌号
license_plate_pattern = re.compile(r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4,5}[A-Z0-9挂学警港澳]{1}')

def clean_chinese_text(text):
    text = text.replace('至', '')  # 删除“至”
    text = text.replace('接', '')  # 删除“接”
    text = text.replace('及', '')  # 删除“及”
    text_without_to = text.replace('冀', '')  # 删除“冀”
    text_without_license = license_plate_pattern.sub('', text_without_to)  # 删除车牌
    text_without_numbers_and_letters = re.sub(r'[^\u4e00-\u9fa5]', '', text_without_license)  # 删除桩号

    tokens = jieba.lcut(text_without_numbers_and_letters, cut_all=False)  # 分词

    # 去除停用词
    tokens = [token for token in tokens if token not in stopwords]

    return " ".join(tokens)


accident_data['description_early1'] = accident_data['description_early1'].apply(clean_chinese_text)
with open('all_sentences.txt', 'w', encoding='utf-8') as f:
    for item in accident_data['description_early1']:
        f.write(item + '\n')
##################################################
categorical_columns = ['Weekday', 'Infrastructure_damage', 'Injury', 'Death', 'Vehicle_type', 'Vehicle_involved',
                      'Pavement_condition', 'Weather_condition', 'Shoulder', 'Burning', 'Rollover', 'Night_hours',
                      'Peak_hours', 'Ramp']
duration = accident_data.pop('duration')
# 划分训练集、验证集与测试集
train_val_data, test_data, train_val_duration, test_duration = train_test_split(accident_data, duration, test_size=0.20, random_state=42, shuffle=True)



# 提取文本数据做单独处理
train_val_text = train_val_data.pop("description_early1")
test_text = test_data.pop("description_early1")

# 文本embedding
vector_size = 128
data_file = 'all_sentences.txt'
word2vec_model = Word2Vec(LineSentence(data_file), vector_size=vector_size, window=10, min_count=5, workers=4, sg=1, epochs=20)
batch_size = 16
def sentence_to_vec(sentence, model, vector_size=vector_size):
    words = sentence
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if not word_vecs:
        return np.zeros(vector_size)  # if words are not in vocab, return zero vector
    return np.mean(word_vecs, axis=0)


  # shape: (num_sentences, vector_size)
test_text_features = [sentence_to_vec(sentence, word2vec_model) for sentence in test_text]
# 定义模型
class TextRegressionNet(nn.Module):
    """
    vacb_size: 词汇表大小，注意不是文本特征维度
    """
    def __init__(self, vocab_size, categorical_features_size, out_features=1):
        super(TextRegressionNet, self).__init__()
        self.n_layers = n_layers = 4  # LSTM的层数
        self.hidden_dim = hidden_dim = 128  # LSTM隐状态的维度
        self.dense_size = dense_size = 256     # 稠密层维度-64，128，256
        dense_size2 = 64  # 稠密层2维度 32，64，128
        embedding_dim = word2vec_model.wv.vector_size  # 嵌入维度为word2vec词向量的维度
        drop_prob = 0.4  # dropout

        # 定义embedding，使用word2vec预训练模型初始化embedding权重
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        #添加layer Normalization层
        #self.layer_norm = nn.LayerNorm(embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,  # 输入的维度
                            hidden_dim,  # LSTM输出的hidden_state的维度
                            n_layers,  # LSTM的层数
                            dropout=drop_prob,
                            bidirectional=False,
                            batch_first=True  # 第一个维度是否是batch_size
                            )
        #self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(dense_size + categorical_features_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.Linear(64, 1),
        )

        # LSTM结束后的全连接线性层
        self.fc1 = nn.Linear(in_features=hidden_dim,  # 将LSTM的输出作为线性层的输入
                            out_features=dense_size
                            )
        self.fc2 = nn.Linear(in_features=dense_size + categorical_features_size,  # 将LSTM的输出作为线性层的输入
                            out_features=out_features
                            )
        self.fc3 = nn.Linear(in_features=dense_size2,
                            out_features=out_features
                            )

        # 给最后的全连接层加一个Dropout
        self.dropout = nn.Dropout(drop_prob)
        self.layer_num = nn.LayerNorm(dense_size + categorical_features_size)

    def forward(self, x, hidden):
        """
        x: 本次的输入，其size为(batch_size, text_feature_size + categorical_features_size)
        hidden: 上一时刻的Hidden State和Cell State。类型为tuple: (h, c),
        其中h和c的size都为(n_layers, batch_size, hidden_dim)
        """
        batch_size = x.size(0)
        text_features = x[:, :vector_size]
        categorical_features = x[:, vector_size:].squeeze(1)
        #print(categorical_features)
        text_features = self.embedding(text_features)  # text做nn.embedding
        #text_features = text_features.permute(1, 0, 2)
        lstm_out, hidden = self.lstm(text_features, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        #lstm_sequence_avg = torch.mean(lstm_out, dim=1)
        #fc_input = lstm_out
        #lstm_out = lstm_out[:,-1,:]
        #lstm_out = self.dropout(lstm_out)
        lstm_dense_out = self.fc1(lstm_out)  # 将LSTM的输出连接至稠密层用以平衡文本与分类特征的数量
        lstm_dense_out = F.relu(lstm_dense_out)
        concat_out = torch.cat((lstm_dense_out, categorical_features), dim=1)
        #lstm_out = torch.cat((lstm_out[:, -1, :], categorical_features), dim=1)
        #out = F.relu(self.fc(lstm_out))
        #normalized_concat = self.layer_num(concat_layer)
        MLP_out = self.out(concat_out)
        #out = self.dropout(dense_out)
        #out = self.fc3(dense_out)

        return MLP_out, hidden

    def init_hidden(self, batch_size):
        """
        初始化隐状态：第一次送给LSTM时，没有隐状态，所以要初始化一个
        这里的初始化策略是全部赋0。
        这里之所以是tuple，是因为LSTM需要接受两个隐状态hidden state和cell state
        """
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
                  )
        return hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextRegressionNet(len(word2vec_model.wv.key_to_index) + 1, train_val_data.shape[1])
model.to(device)

criterion = nn.MSELoss()
lr = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
epochs = 20
patience = 5  # 早停epoch设置
no_improvement_count = 0

def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in data_loader:

        hidden = model.init_hidden(batch_size)
        inputs, labels = batch
        inputs = torch.tensor(inputs, dtype=torch.long).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        outputs, new_hidden = model(inputs, hidden)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(data_loader.dataset)
def val_epoch(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            hidden = model.init_hidden(batch_size)
            inputs, labels = batch
            inputs = torch.tensor(inputs, dtype=torch.long).to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            outputs, new_hidden = model(inputs, hidden)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
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

    train_fold_duration_np = train_fold_duration.values
    val_fold_duration_np = val_fold_duration.values
    train_fold_duration_tensor = torch.tensor(train_fold_duration_np, dtype=torch.float32)  # 再转换为Tensor
    val_fold_duration_tensor = torch.tensor(val_fold_duration_np, dtype=torch.float32)


    train_text_features = [sentence_to_vec(sentence, word2vec_model) for sentence in train_fold_text]
    val_text_features = [sentence_to_vec(sentence, word2vec_model) for sentence in val_fold_text]

    train_combined_features = np.concatenate((train_text_features, train_fold_data), axis=1)
    val_combined_features = np.concatenate((val_text_features, val_fold_data), axis=1)

    train_loader = DataLoader(TensorDataset(torch.tensor(train_combined_features), train_fold_duration_tensor),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_combined_features), val_fold_duration_tensor),
                            batch_size=batch_size, shuffle=False, drop_last=True)

    # 标签处理（归一化or lg变换）
#     scaler = MinMaxScaler()
#     train_duration_norm = scaler.fit_transform(train_fold_duration.values.reshape(-1, 1))
#     val_duration_norm = scaler.transform(val_fold_duration.values.reshape(-1, 1))
#     train_fold_duration = train_duration_norm.squeeze()
#     val_fold_duration = val_duration_norm.squeeze()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion)
        with torch.no_grad():
            val_loss = val_epoch(model, val_loader, criterion)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()

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
torch.save(model.state_dict(), 'lstm_mlp_10fold.pth')

print("Best model across all folds saved.")


# 前200维为文本特征，后14维为分类特征
test_combined_features = np.concatenate((test_text_features, test_data), axis=1)

# 构建数据集
test_loader = DataLoader(TensorDataset(torch.tensor(test_combined_features), torch.tensor(test_duration)), batch_size=batch_size, shuffle=False, drop_last=True)

# 将Word2Vec的词向量加载到nn.Embedding层
# for i in range(len(word2vec_model.wv.key_to_index)):
#     model.embedding.weight.data[i] = torch.from_numpy(word2vec_model.wv[word2vec_model.wv.index_to_key[i]])
# # 在训练过程中不更新embedding权重，设置requires_grad=False
# model.embedding.weight.requires_grad_(False)

# # 绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# def calculate_metrics(predictions, targets):
#     rmse = 1/2 * np.sqrt(mean_squared_error(targets, predictions))
#     mae = 1/2 * mean_absolute_error(targets, predictions)
#     mape = 1/2 * np.mean(np.abs((targets - predictions) / targets)) * 100  # 注意处理除零的情况
#     return rmse, mae, mape
#
# time_intervals = [(0,30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 182)]
#
# # 加载最佳模型权重
# print('----------loading model---------------')
# model.load_state_dict(torch.load('lstm_mlp_model.pth'))
# model.eval()
#
# test_preds = []
# test_labels = []
#
# with torch.no_grad():
#     for batch in test_loader:
#         hidden = model.init_hidden(batch_size)
#         inputs, labels = batch
#         inputs = torch.tensor(inputs, dtype=torch.long).to(device)
#         labels = torch.tensor(labels, dtype=torch.float32).to(device)
#         outputs, new_hidden = model(inputs, hidden)
#         test_preds.extend(outputs.cpu().detach().numpy())
#         test_labels.extend(labels.cpu().detach().numpy())
#
# # 转换为NumPy数组
# test_preds = np.array(test_preds)
# test_labels = np.array(test_labels)
# predictions_original_scale = scaler.inverse_transform(test_preds.reshape(-1, 1))
# actuals_original_scale = scaler.inverse_transform(test_labels.reshape(-1, 1))
#
# # 计算指标
# rmse, mae, mape = calculate_metrics(predictions_original_scale, actuals_original_scale)
# print(f"Test Set Metrics: RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}%")
#
# interval_results = []
# for interval in time_intervals:
#     interval_mask = (actuals_original_scale >= interval[0]) & (actuals_original_scale <= interval[1])
#     interval_preds = predictions_original_scale[interval_mask]
#     interval_actuals = actuals_original_scale[interval_mask]
#
#     if not interval_preds.size or not interval_actuals.size:
#         print(f"No data in the interval {interval[0]}-{interval[1]} minutes.")
#         continue
#
#     rmse, mae, mape = calculate_metrics(interval_preds, interval_actuals)
#     interval_results.append({
#         "Interval": f"{interval[0]}-{interval[1]} min",
#         "RMSE": rmse,
#         "MAE": mae,
#         "MAPE": mape
#     })
#
# # 创建DataFrame并保存到CSV文件
# df = pd.DataFrame(interval_results)
# df.to_csv("interval_metrics_min10_lstm_mlp_21.csv", index=False)

#print("Metrics saved to 'interval_metrics.csv'")