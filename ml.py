import pandas as pd
import numpy as np
from chinese_calendar import is_workday  # 工作日判断
import matplotlib
import matplotlib.pyplot as plt  # 绘图库
matplotlib.use('TkAgg')
import jieba  # 中文分词
import re  # 正则化
import os  # 读取文件

from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.preprocessing import MinMaxScaler  # 归一化处理
from gensim.models import Word2Vec  # 词嵌入
from gensim.models.word2vec import LineSentence
import shap

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



pd.set_option('display.max_columns', None)  # 显示全部列

df = pd.read_csv('accident_data_new1.csv',encoding='gbk')
# 删除多列缺失值
columns_to_check = ['location_type', 'weather', 'environment_condition', 'vehicle', 'impact', 'death_num', 'injury_num', 'duration_h', 'description']
df = df.dropna(subset=columns_to_check, how='any', axis=0)
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
    # text = text.replace('至', '')  # 删除“至”
    # text = text.replace('接', '')  # 删除“接”
    # text = text.replace('及', '')  # 删除“及”
    text_without_to = text.replace('冀', '')  # 删除“冀”
    text_without_license = license_plate_pattern.sub('', text_without_to)  # 删除车牌
    #text_without_numbers_and_letters = re.sub(r'[^\u4e00-\u9fa5]', '', text_without_license)  # 删除桩号

    tokens = jieba.lcut(text_without_license, cut_all=False)  # 分词

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
train_val_data, test_data, train_val_duration, test_duration = train_test_split(accident_data, duration, test_size=0.15, random_state=42, shuffle=True)
#train_data, val_data, train_duration, val_duration = train_test_split(train_val_data, train_val_duration, test_size=0.15, random_state=42, shuffle=True)


# 归一化duration
scaler = MinMaxScaler()
train_duration_norm = scaler.fit_transform(train_val_duration.values.reshape(-1, 1))
#val_duration_norm = scaler.transform(val_duration.values.reshape(-1, 1))
test_duration_norm = scaler.transform(test_duration.values.reshape(-1, 1))
train_duration = train_duration_norm.squeeze()
#val_duration = val_duration_norm.squeeze()
test_duration = test_duration_norm.squeeze()

# 提取文本数据做单独处理
train_text = train_val_data.pop('description_early1')
#val_text = val_data.pop('description_early1')
test_text = test_data.pop('description_early1')
# with open('train_sentences.txt', 'w', encoding='utf-8') as f:
#     for sentence in train_text:
#         f.write(''.join(sentence) + '\n')
# print("111111111111111111111111")
# feature_name = train_data_onehot.columns.tolist()
# #绘制词云图
# all_descriptions_str = ' '.join(train_text)
#
# # 创建词云对象
# wordcloud = WordCloud(font_path=r'C:\Windows\Fonts\simhei.ttf', width=800, height=600, max_words=100, background_color='white',collocations=False, min_font_size=6).generate(all_descriptions_str)
#
# # 显示词云图
# plt.figure(figsize=(10, 8))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.savefig('wordcloud.png', dpi=300)
# plt.show()

# 文本embedding

# class SentenceIterator:
#     def __init__(self, sentences):
#         self.sentences = sentences
#
#     def __iter__(self):
#         for sentence in self.sentences:
#             yield ''.join(sentence)
# train_sentences_iter = SentenceIterator(train_text)
#
# model = Word2Vec(sentences=train_sentences_iter, vector_size=200, window=5, min_count=1, workers=4, sg=1)
# for index, key in enumerate(model.wv.index_to_key[:10]):
#     count = model.wv.get_vecattr(key, 'count')
#     print(f"{key}: {count}")
data_file = 'all_sentences.txt'
model = Word2Vec(LineSentence(data_file), vector_size=128, window=5, min_count=1, workers=4, sg=1)
# with open('word_vectors.txt', 'w', encoding='utf-8') as f:
#     for word in model.wv.key_to_index:
#         vector = ' '.join(map(str, model.wv[word]))
#         f.write(f'{word} {vector}\n')
# result = model.wv.most_similar('行车道')
# print(result)
# 定义函数将句子转换为词嵌入向量
def sentence_to_vec(sentence, model, vector_size=128):
    words = sentence
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if not word_vecs:
        return np.zeros(vector_size)  # if words are not in vocab, return zero vector
    return np.mean(word_vecs, axis=0)


train_text_features = [sentence_to_vec(sentence, model) for sentence in train_text]  # shape: (num_sentences, vector_size)
#val_text_features = [sentence_to_vec(sentence, model) for sentence in val_text]
test_text_features = [sentence_to_vec(sentence, model) for sentence in test_text]

# 对文本特征进行归一化，避免与分类特征的量级差距（无需做归一化，文本编码后的特征已完成）
# scaler1 = MinMaxScaler()
# train_text_features_normalized = scaler1.fit_transform(train_text_features)
# val_text_features_normalized = scaler1.transform(val_text_features)
# test_text_features_normalized = scaler1.transform(test_text_features)
train_text_features = np.array(train_text_features)
#val_text_features = np.array(val_text_features)
test_text_features = np.array(test_text_features)
# 输出维度
#train_text_features_array = np.array(train_text_features)
#print(train_text_features_array.shape)

# method1: 先拼接特征，再训练
assert train_val_data.shape[0] == train_text_features.shape[0]

train_combined_features = np.concatenate((train_val_data, train_text_features), axis=1)  # shape: (num_samples, Categorical features+Text features)
#val_combined_features = np.concatenate((val_data, val_text_features), axis=1)
test_combined_features = np.concatenate((test_data, test_text_features), axis=1)

# 构建模型
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from sklearn.model_selection import cross_val_score

# # 随机森林
# rf_model = RandomForestRegressor(n_estimators=200, random_state=42)  # 可以调整n_estimators的值
# rf_model.fit(train_combined_features, train_duration)
# y_pred_rf = rf_model.predict(test_combined_features)
#
#
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_rf).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))
# rmse_original_scale = np.sqrt(mean_squared_error(test_duration_original_scaler, y_pred_original_scale.flatten()))
# mae = mean_absolute_error(test_duration_original_scaler, y_pred_original_scale.flatten())
# MAPE = np.mean(np.abs((test_duration_original_scaler - y_pred_original_scale.flatten()) / test_duration_original_scaler)) * 100
# print(f"Test Mean Squared Error in Original Scale of RF: {rmse_original_scale}")
# print(f"Test Mean Absolute Error: {mae}")
# print(f"Test Mean Absolute Percentage Error: {MAPE}%")

# #GBDT
# gbt_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
#
# # 使用梯度提升回归树模型拟合数据
# gbt_model.fit(train_combined_features, train_duration)
#
# # 使用模型进行预测
# y_pred_gbt = gbt_model.predict(test_combined_features)
#
# # 将预测结果和测试集的目标值转换回原始尺度
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_gbt).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))
#
# # 计算评估指标
# rmse_original_scale = np.sqrt(mean_squared_error(test_duration_original_scale, y_pred_original_scale.flatten()))
# mae = mean_absolute_error(test_duration_original_scale, y_pred_original_scale.flatten())
# MAPE = np.mean(np.abs((test_duration_original_scale - y_pred_original_scale.flatten()) / test_duration_original_scale)) * 100
#
# # 打印评估结果
# print(f"Test R Mean Squared Error in Original Scale of GBT: {rmse_original_scale}")
# print(f"Test Mean Absolute Error: {mae}")
# print(f"Test Mean Absolute Percentage Error: {MAPE}%")

# # XGBOOST
# xgb_model = xgb.XGBRegressor(n_estimators=200, random_state=42)
# scores = cross_val_score(xgb_model, train_combined_features, train_duration, cv=10, scoring='neg_mean_squared_error')
# mean_neg_mse = np.mean(scores)
#
# rmse_cv = np.sqrt(-mean_neg_mse)
# print(f"十次交叉验证的平均均方根误差 (RMSE): {rmse_cv}")
# xgb_model.fit(train_combined_features, train_duration)
# y_pred_xgb = xgb_model.predict(test_combined_features)
#
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_xgb).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))
# rmse_original_scale = np.sqrt(mean_squared_error(test_duration_original_scaler, y_pred_original_scale.flatten()))
# mae = mean_absolute_error(test_duration_original_scaler, y_pred_original_scale.flatten())
# MAPE = np.mean(np.abs((test_duration_original_scaler - y_pred_original_scale.flatten()) / test_duration_original_scaler)) * 100
# print(f"Test Mean Absolute Error: {mae}")
# print(f"Test Mean Absolute Percentage Error: {MAPE}%")
# print(f"R Test Mean Squared Error in Original Scale of XGB: {rmse_original_scale}")



# # lightgbm
# lgb_model = lgb.LGBMRegressor(n_estimators=200, random_state=42)
# scores = cross_val_score(lgb_model, train_combined_features, train_duration, cv=10, scoring='neg_mean_squared_error')
# mean_neg_mse = np.mean(scores)
#
# rmse_cv = np.sqrt(-mean_neg_mse)
# print(f"十次交叉验证的平均均方根误差 (RMSE): {rmse_cv}")
#
#
# lgb_model.fit(train_combined_features, train_duration)
# y_pred_lgb = lgb_model.predict(test_combined_features)
#
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_lgb).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))
# rmse_original_scale = np.sqrt(mean_squared_error(test_duration_original_scale, y_pred_original_scale.flatten()))
# mae = mean_absolute_error(test_duration_original_scale, y_pred_original_scale.flatten())
# MAPE = np.mean(np.abs((test_duration_original_scale - y_pred_original_scale.flatten()) / test_duration_original_scale)) * 100
# print(f"Test Mean Absolute Error: {mae}")
# print(f"Test Mean Absolute Percentage Error: {MAPE}%")
# print(f"R Test Mean Squared Error in Original Scale of XGB: {rmse_original_scale}")

#
# from matplotlib import rcParams
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#
# # explainer = shap.Explainer(lgb_model,test_data_onehot)
# #
# # shap_values = explainer(test_data_onehot)
# # # 使用排序后的数据绘制summary_plot
# # shap.plots.heatmap(shap_values, max_display=13, show=False)
# # plt.xlabel("Instances", fontsize=13)
# # #plt.show()
# # plt.savefig('shap_heatmap_plot.svg', format='svg', bbox_inches='tight', dpi = 1200)
#
#
# # summary_plot
# explainer = shap.Explainer(lgb_model, test_data_onehot)
# shap_values = explainer(test_data_onehot)
# shap.plots.beeswarm(shap_values, max_display=13, show=False)
# #plt.show()
# plt.savefig('shap_summary_plot_vector.svg', format='svg', bbox_inches='tight', dpi = 1200)


# #catboost
# catboost_model = CatBoostRegressor(iterations=200, random_state=42, loss_function='RMSE')  # 使用RMSE损失函数
#
# # 注意：CatBoost不需要显式调用fit，可以直接传入分类特征和目标变量
# catboost_model.fit(train_combined_features, train_duration, eval_set=(test_combined_features, test_duration))
#
# y_pred_catboost = catboost_model.predict(test_combined_features)
#
#
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_catboost).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))
# rmse_original_scale = np.sqrt(mean_squared_error(test_duration_original_scale, y_pred_original_scale.flatten()))
# mae = mean_absolute_error(test_duration_original_scale, y_pred_original_scale.flatten())
# MAPE = np.mean(np.abs((test_duration_original_scale - y_pred_original_scale.flatten()) / test_duration_original_scale)) * 100
# print(f"Test Mean Absolute Error: {mae}")
# print(f"Test Mean Absolute Percentage Error: {MAPE}%")
# print(f"R Test Mean Squared Error in Original Scale of XGB: {rmse_original_scale}")

#SVR

svr_model = SVR(kernel='rbf', C=1e3, gamma='auto')  # 可以调整C和gamma的值
svr_model.fit(train_val_data, train_duration)
y_pred_svr = svr_model.predict(test_data)
#
# # mse = mean_squared_error(test_duration, y_pred_svr)
# # mae = mean_absolute_error(test_duration, y_pred_svr)
# # r2 = r2_score(test_duration, y_pred_svr)
#
# # print(f"Test Mean Squared Error: {mse}")
# # print(f"Test R² Score: {r2}")
#
y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_svr).reshape(-1, 1))
test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))
# rmse_original_scale = np.sqrt(mean_squared_error(actuals_original_scale, predictions_original_scale.flatten()))
# mae = mean_absolute_error(actuals_original_scale, predictions_original_scale.flatten())
# MAPE = np.mean(np.abs((actuals_original_scale - predictions_original_scale.flatten()) / actuals_original_scale)) * 100
# print(f"Test Mean Squared Error in Original Scale of SVR: {rmse_original_scale}")
# print(f"Test Mean Absolute Error: {mae}")
# print(f"Test Mean Absolute Percentage Error: {MAPE}%")



def calculate_metrics(predictions, targets):
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100  # 注意处理除零的情况
    return rmse, mae, mape

time_intervals = [(0,30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 185)]

rmse, mae, mape = calculate_metrics(y_pred_original_scale, test_duration_original_scale)
print(f"Test Set Metrics: RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}%")
# #
# interval_results = []
# for interval in time_intervals:
#     interval_mask = (test_duration_original_scale >= interval[0]) & (test_duration_original_scale <= interval[1])
#     interval_preds = y_pred_original_scale[interval_mask]
#     interval_actuals = test_duration_original_scale[interval_mask]
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
# df = pd.DataFrame(interval_results)
# df.to_csv("CatBoost_group_metrics.csv", index=False)

