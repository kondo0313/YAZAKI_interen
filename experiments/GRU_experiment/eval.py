import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd())

import models.Preprocessing as pre

# 推論に必要なパラメータの設定
today = '2024-05-04'
week_1 = 7
week_2 = 14
week_3 = 21
output_len = 50

data_path = '/home/code/data/processed/week_dataset.csv'
## calenderは使わないかも
# calender_path = '/home/code/data/calender_ver2.csv'

feature_path1 = '/home/code/artifacts/models/GRU_latest/1week/GRU_features.csv'
feature_path2 = '/home/code/artifacts/models/GRU_latest/2week/GRU_features.csv'
feature_path3 = '/home/code/artifacts/models/GRU_latest/3week/GRU_features.csv'
model_path1 = '/home/code/artifacts/models/GRU_latest/1week/gru_best_model_gru512_dense1024_loss0.0045.h5'
model_path2 = '/home/code/artifacts/models/GRU_latest/2week/gru_best_model_gru256_dense1024_loss0.0040.h5'
model_path3 = '/home/code/artifacts/models/GRU_latest/3week/gru_best_model_gru512_dense1024_loss0.0043.h5'

output_path = f'/home/code/artifacts/models/GRU_latest/pred_GRU_{today}.csv'

weekly_total_columns = ['Shipment', 'Teisyoku', 'Temp', 'High Temp', 'Low Temp', 'Status']
weekly_mean_columns = ['Week Shipment']
shift_unchange_name_columns = ['Status', 'Week Status', 'Week Shipment Mean', 'Week Status2']
shift_change_name_columns = ['Temp', 'Week Temp']
env_columns1 = ['Temp-'+str(week_1), 'Week Temp-'+str(week_1)]
env_columns2 = ['Temp-'+str(week_2), 'Week Temp-'+str(week_2)]
env_columns3 = ['Temp-'+str(week_3), 'Week Temp-'+str(week_3)]
cat_columns = ['Week Number']

drop_columns = ['Target', 'Week', 'Week Start', 'Week End', 'Week Status2', 'Status', 'Teisyoku', 'Week Teisyoku']

# データの読み込み
df = pd.read_csv(data_path)

# df_calender = pd.read_csv(calender_path, index_col=0, parse_dates=True)

df_week1 = df.copy()
df_week2 = df.copy()
df_week3 = df.copy()

df_week1 = (
        df_week1.pipe(pre.change_column_name)
        .pipe(pre.set_index_date)
#         .pipe(pre.add_pred_rows, today=today, week_number_li=week_number_li, temp_li=temp_li, target_days=week_1)
#         .pipe(pre.add_status, today=today, target_days=week_1, calender_path=calender_path)
#         .pipe(pre.add_weekday)
#         .pipe(pre.add_weekly_total, columns=weekly_total_columns)
#         .pipe(pre.add_weekly_mean, columns=weekly_mean_columns) 
#         .pipe(pre.add_week_status2)
#         .pipe(pre.shift_unchange_name, shift_columns=shift_unchange_name_columns, shift_days=week_1)
#         .pipe(pre.shift_change_name, shift_columns=shift_change_name_columns, shift_days=week_1)
#         .pipe(pre.change_env_data, env_columns=env_columns1)
#         .pipe(pre.add_target, target_days=week_1) # Week Shipmentをズラす前に作成することに注意
#         .pipe(pre.add_target2, target_days=week_1) # Week Shipmentをズラす前に作成することに注意
#         .pipe(pre.shift_unchange_name, shift_columns=['Week Shipment'], shift_days=-7) # Week Shipmentは1週間前のデータを使用
#         .pipe(pre.categorize_columns, cat_columns=cat_columns)
)
print(df_week1)

# df_week2 = (
#         df_week2.pipe(pre.change_column_name)
#         .pipe(pre.set_index_date)
#         .pipe(pre.add_pred_rows, today=today, week_number_li=week_number_li, temp_li=temp_li, target_days=week_2)
#         .pipe(pre.add_status, today=today, target_days=week_2, calender_path=calender_path)
#         .pipe(pre.add_weekday)
#         .pipe(pre.add_weekly_total, columns=weekly_total_columns)
#         .pipe(pre.add_weekly_mean, columns=weekly_mean_columns) 
#         .pipe(pre.add_week_status2)
#         .pipe(pre.shift_unchange_name, shift_columns=shift_unchange_name_columns, shift_days=week_2)
#         .pipe(pre.shift_change_name, shift_columns=shift_change_name_columns, shift_days=week_2)
#         .pipe(pre.change_env_data, env_columns=env_columns2)
#         .pipe(pre.add_target, target_days=week_2) # Week Shipmentをズラす前に作成することに注意
#         .pipe(pre.add_target2, target_days=week_2) # Week Shipmentをズラす前に作成することに注意
#         .pipe(pre.shift_unchange_name, shift_columns=['Week Shipment'], shift_days=-7) # Week Shipmentは1週間前のデータを使用
#         .pipe(pre.categorize_columns, cat_columns=cat_columns)
# )

# df_week3 = (
#         df_week3.pipe(pre.change_column_name)
#         .pipe(pre.set_index_date)
#         .pipe(pre.add_pred_rows, today=today, week_number_li=week_number_li, temp_li=temp_li, target_days=week_3)
#         .pipe(pre.add_status, today=today, target_days=week_3, calender_path=calender_path)
#         .pipe(pre.add_weekday)
#         .pipe(pre.add_weekly_total, columns=weekly_total_columns)
#         .pipe(pre.add_weekly_mean, columns=weekly_mean_columns) 
#         .pipe(pre.add_week_status2)
#         .pipe(pre.shift_unchange_name, shift_columns=shift_unchange_name_columns, shift_days=week_3)
#         .pipe(pre.shift_change_name, shift_columns=shift_change_name_columns, shift_days=week_3)
#         .pipe(pre.change_env_data, env_columns=env_columns3)
#         .pipe(pre.add_target, target_days=week_3) # Week Shipmentをズラす前に作成することに注意
#         .pipe(pre.add_target2, target_days=week_3) # Week Shipmentをズラす前に作成することに注意
#         .pipe(pre.shift_unchange_name, shift_columns=['Week Shipment'], shift_days=-7) # Week Shipmentは1週間前のデータを使用
#         .pipe(pre.categorize_columns, cat_columns=cat_columns)
# )