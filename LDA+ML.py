import pandas as pd
import numpy as np
from chinese_calendar import is_workday  # 工作日判断
import matplotlib
matplotlib.use('TkAgg')
import jieba  # 中文分词
import re  # 正则化
import os  # 读取文件

from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.preprocessing import MinMaxScaler  # 归一化处理
import shap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

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
upper_bound = Q3 + 1.5 * IQR   #TODO:change the threshold

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
    # text_without_to = text.replace('冀', '')  # 删除“冀”
    # text_without_license = license_plate_pattern.sub('', text_without_to)  # 删除车牌
    # text_without_numbers_and_letters = re.sub(r'[^\u4e00-\u9fa5]', '', text_without_license)  # 删除桩号
    #
    tokens = jieba.lcut(text, cut_all=False)  # 分词

    # 去除停用词
    tokens = [token for token in tokens if token not in stopwords]

    return " ".join(tokens)
#accident_data['description_early1'] = accident_data['description_early1'].apply(clean_chinese_text)


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
accident_data['description'] = accident_data['description'].apply(process_text_with_times)
accident_data['description'] = accident_data['description'].apply(clean_chinese_text)

with open('all_sentences.txt', 'w', encoding='utf-8') as f:
    for item in accident_data['description']:
        f.write(item + '\n')
##################################################
categorical_columns = ['Weekday', 'Infrastructure_damage', 'Injury', 'Death', 'Vehicle_type', 'Vehicle_involved',
                      'Pavement_condition', 'Weather_condition', 'Shoulder', 'Burning', 'Rollover', 'Night_hours',
                      'Peak_hours', 'Ramp']
duration = accident_data.pop('duration')

# 划分训练集、验证集与测试集
train_val_data, test_data, train_val_duration, test_duration = train_test_split(accident_data, duration, test_size=0.20, random_state=42, shuffle=True)

# 归一化duration
# scaler = MinMaxScaler()
# train_duration_norm = scaler.fit_transform(train_val_duration.values.reshape(-1, 1))
# test_duration_norm = scaler.transform(test_duration.values.reshape(-1, 1))
# train_duration = train_duration_norm.squeeze()
# test_duration = test_duration_norm.squeeze()

#对标签进行对数变换，使其近似正态分布
train_duration_log = np.log(train_val_duration.values)
test_duration_log = np.log(test_duration.values)
train_duration = train_duration_log
test_duration = test_duration_log

# 提取文本数据做单独处理
train_text = train_val_data.pop('description')
test_text = test_data.pop('description')

vectorizer = CountVectorizer(max_df=0.5, min_df=2, max_features=1000, stop_words='english')
dtm = vectorizer.fit_transform(train_text)
test_dtm = vectorizer.transform(test_text)
feature_names = vectorizer.get_feature_names_out()

num_topics = 30
alpha = 0.5
eta = 0.5
lda = LatentDirichletAllocation(
    n_components=num_topics,
    doc_topic_prior=alpha,
    topic_word_prior=eta,
    random_state=42)
lda_Z = lda.fit_transform(dtm)
test_lda_Z = lda.transform(test_dtm)
topic_word_distributions = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
n_top_words = 6
# for topic_idx, topic in enumerate(topic_word_distributions):
#     top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
#     top_features = [feature_names[i] for i in top_features_ind]
#     weights = topic[top_features_ind]
#
#     print(f"tp #{topic_idx}: ", end="")
#     output = ' + '.join([f"{weight:.2f}⁎{feature}" for weight, feature in zip(weights, top_features)])
#     print(output)


lda_df = pd.DataFrame(lda_Z, columns=[f'topic_{i}' for i in range(num_topics)], index=train_text.index)
test_lda_df = pd.DataFrame(test_lda_Z, columns=[f'topic_{i}' for i in range(num_topics)], index=test_text.index)
assert set(lda_df.index) == set(train_val_data.index), "lda_df and train_val_data must have the same index"
train_combined_data = pd.concat([train_val_data, lda_df], axis=1)
test_combined_data = pd.concat([test_data, test_lda_df], axis=1)


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
# data_file = 'all_sentences.txt'
# model = Word2Vec(LineSentence(data_file), vector_size=128, window=5, min_count=1, workers=4, sg=1)
# # with open('word_vectors.txt', 'w', encoding='utf-8') as f:
# #     for word in model.wv.key_to_index:
# #         vector = ' '.join(map(str, model.wv[word]))
# #         f.write(f'{word} {vector}\n')
# # result = model.wv.most_similar('行车道')
# # print(result)
# # 定义函数将句子转换为词嵌入向量
# def sentence_to_vec(sentence, model, vector_size=128):
#     words = sentence
#     word_vecs = [model.wv[word] for word in words if word in model.wv]
#     if not word_vecs:
#         return np.zeros(vector_size)  # if words are not in vocab, return zero vector
#     return np.mean(word_vecs, axis=0)
#
#
# train_text_features = [sentence_to_vec(sentence, model) for sentence in train_text]  # shape: (num_sentences, vector_size)
# #val_text_features = [sentence_to_vec(sentence, model) for sentence in val_text]
# test_text_features = [sentence_to_vec(sentence, model) for sentence in test_text]
#
# # 输出维度
# #train_text_features_array = np.array(train_text_features)
# #print(train_text_features_array.shape)
#
# # method1: 先拼接特征，再训练
# assert train_val_data.shape[0] == train_text_features.shape[0]
#
# train_combined_features = np.concatenate((train_val_data, train_text_features), axis=1)  # shape: (num_samples, Categorical features+Text features)
# #val_combined_features = np.concatenate((val_data, val_text_features), axis=1)
# test_combined_features = np.concatenate((test_data, test_text_features), axis=1)

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
# rf_model = RandomForestRegressor(n_estimators=300, random_state=42)  # 可以调整n_estimators的值
# rf_model.fit(train_combined_data, train_duration)
# y_pred_rf = rf_model.predict(test_combined_data)
#
#
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_rf).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))


# #GBDT
# gbt_model = GradientBoostingRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, min_samples_split=4, learning_rate=0.01,random_state=42)
# # 使用梯度提升回归树模型拟合数据
# gbt_model.fit(train_combined_data, train_duration)
#
# # 使用模型进行预测
# y_pred_gbt = gbt_model.predict(test_combined_data)
#
# # 将预测结果和测试集的目标值转换回原始尺度
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_gbt).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))


# from sklearn.model_selection import cross_val_predict
# gbt_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
# y_pred_gbt = cross_val_predict(gbt_model, train_combined_data, train_duration, cv=10)
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_gbt).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(train_duration.reshape(-1, 1))


# # XGBOOST
# xgb_model = xgb.XGBRegressor(n_estimators=200, random_state=42)
# # scores = cross_val_score(xgb_model, train_combined_data, train_duration, cv=10, scoring='neg_mean_squared_error')
# # mean_neg_mse = np.mean(scores)
# #
# # rmse_cv = np.sqrt(-mean_neg_mse)
# # print(f"十次交叉验证的平均均方根误差 (RMSE): {rmse_cv}")
# xgb_model.fit(train_combined_data, train_duration)
# y_pred_xgb = xgb_model.predict(test_combined_data)
#
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_xgb).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))


# lightgbm
lgb_model = lgb.LGBMRegressor(n_estimators=200, random_state=42)
# scores = cross_val_score(lgb_model, train_combined_data, train_duration, cv=10, scoring='neg_mean_squared_error')
# mean_neg_mse = np.mean(scores)
#
# rmse_cv = np.sqrt(-mean_neg_mse)
# print(f"十次交叉验证的平均均方根误差 (RMSE): {rmse_cv}")


lgb_model.fit(train_val_data, train_val_duration)
# y_pred_lgb = lgb_model.predict(test_data)
#
# # y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_lgb).reshape(-1, 1))
# # test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))
#
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# #
# explainer = shap.Explainer(lgb_model,test_data)
#
# shap_values = explainer(test_data)
# # 使用排序后的数据绘制summary_plot
# shap.plots.heatmap(shap_values, max_display=14, show=False)
# plt.xlabel("Instances", fontsize=13)
# #plt.show()
# plt.savefig('shap_heatmap_plot1.svg', format='svg', bbox_inches='tight', dpi = 1200)
#
#
# # summary_plot
# explainer = shap.Explainer(lgb_model, test_data)
# shap_values = explainer(test_data)
# shap.plots.beeswarm(shap_values, max_display=14, show=False)
# #plt.show()
# plt.savefig('shap_summary_plot_vector1.svg', format='svg', bbox_inches='tight', dpi = 1200)

#PDP
common_params = {
    "subsample": 100,
    "n_jobs": 1,
    "grid_resolution": 20,
    "random_state": 0,
    #"is_categorical": []
}
print("Computing partial dependence plots and individual conditional expectation...")
#_, ax = plt.subplots(ncols=2, figsize=(6, 4), constrained_layout=True)

features_info = {
    "features":  ['Vehicle_type', 'Rollover', 'Infrastructure_damage', 'Night_hours', 'Weather_condition', 'Shoulder',  'Vehicle_involved', 'Injury', 'Burning', 'Peak_hours','Pavement_condition', 'Death'],
    "kind": "average",
    'categorical_features': categorical_columns,
}
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(7, 7), constrained_layout=True)
display = PartialDependenceDisplay.from_estimator(
    lgb_model,
    test_data,
    **features_info,
    ax=ax,
    **common_params,
)
#_ = display.figure_.suptitle("ICE and PDP representations", fontsize=16)

# Add text annotations to each subplot
for i, axi in enumerate(ax.flatten()):
    # Get the PDP values by calling the method
    _, pdp_values = display.pd_results[i].values()

    # Print the structure of the data for debugging
    print(f"PDP Values for feature {i}: {pdp_values}\n")

    # Iterate over each class and add text annotations
    for j, pdp_value in enumerate(pdp_values):
        # Ensure pdp_value is an array
        if not isinstance(pdp_value, np.ndarray):
            pdp_value = np.array([pdp_value])

        # Calculate the position for the text annotation
        # Find the maximum value of the bar chart
        max_value = np.max(pdp_value)

        # Add text annotation with PDP value above each bar
        for k, single_pdp_value in enumerate(pdp_value):
            # Calculate the position for the text annotation
            x_pos = k  # Use the index as the x-position
            y_pos = single_pdp_value + 0.01  # Offset slightly above the bar

            # Add text annotation with PDP value
            axi.text(x_pos, y_pos, f"{single_pdp_value:.2f}",
                     transform=axi.transData,
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     fontsize=10,
                     color='black')


# Set the y-axis label for all subplots
for i in range(3):  # Assuming 3 rows
    ax[i, 0].set_ylabel('Average effect on duration (min)')

plt.tight_layout(rect=[0, 0, 1, 1])

plt.savefig("pdp_plots2.svg", format="svg", bbox_inches='tight', dpi=1200)
plt.show()
# plt.tight_layout()
#
# plt.savefig("pdp_plots2.svg", format="svg", bbox_inches='tight', dpi = 1200)
# plt.show()


# #catboost
# catboost_model = CatBoostRegressor(iterations=200, random_state=42, loss_function='RMSE')  # 使用RMSE损失函数
#
# catboost_model.fit(train_combined_data, train_duration, eval_set=(test_combined_data, test_duration))
#
# y_pred_catboost = catboost_model.predict(test_combined_data)
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_catboost).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))

# #SVR
# svr_model = SVR(kernel='rbf', C=1e3, gamma='auto')  # 可以调整C和gamma的值
# svr_model.fit(train_combined_data, train_duration)
# y_pred_svr = svr_model.predict(test_combined_data)
#
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_svr).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))

# #MLP
# from sklearn.neural_network import MLPRegressor
#
# # 创建MLP模型实例，两层隐藏层，
# mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500, random_state=42)
#
# # 训练模型
# mlp_model.fit(lda_df, train_duration)
#
# # 预测
# y_pred_mlp = mlp_model.predict(test_lda_df)
#
# # 将预测结果转换回原始尺度
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_mlp).reshape(-1, 1))
#
# # 将测试数据的目标值转换回原始尺度
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))

# #LR
# from sklearn.linear_model import LinearRegression
#
# # 创建线性回归模型实例
# lr_model = LinearRegression()
#
# # 训练模型
# lr_model.fit(train_combined_data, train_duration)
#
# # 预测
# y_pred_lr = lr_model.predict(test_combined_data)
#
# # 将预测结果转换回原始尺度
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_lr).reshape(-1, 1))
#
# # 将测试数据的目标值转换回原始尺度
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))

# #DT
# from sklearn.tree import DecisionTreeRegressor
#
# # 决策树模型
# dtree_model = DecisionTreeRegressor(random_state=42)  # 可以调整参数如max_depth, min_samples_split等
# dtree_model.fit(train_combined_data, train_duration)
# y_pred_dtree = dtree_model.predict(test_combined_data)
#
# # 将预测结果和测试集的目标值转换回原始尺度
# y_pred_original_scale = np.exp(y_pred_dtree)
# test_duration_original_scale = np.exp(test_duration)
# y_pred_original_scale = scaler.inverse_transform(np.array(y_pred_dtree).reshape(-1, 1))
# test_duration_original_scale = scaler.inverse_transform(test_duration.reshape(-1, 1))




# def calculate_metrics(predictions, targets):
#     rmse = np.sqrt(mean_squared_error(targets, predictions))
#     mae = mean_absolute_error(targets, predictions)
#     mape = np.mean(np.abs((targets - predictions) / targets)) * 100  # 注意处理除零的情况
#     return rmse, mae, mape
#
# time_intervals = [(0,30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 185)]
# #
# rmse, mae, mape = calculate_metrics(y_pred_original_scale, test_duration_original_scale)
# print(f"Test Set Metrics: RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}%")
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
# df.to_csv("group_metrics/RF_group_metrics.csv", index=False)

