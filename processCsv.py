import pandas as pd
import os

current_dir = os.path.dirname(__file__)
# 原始数据文件路径
old_csv_path = os.path.join(current_dir, 'models', 'monatszahlen2412_verkehrsunfaelle_06_12_24.csv')
new_csv_path = os.path.join(current_dir, 'models', 'processed_data.csv')

df = pd.read_csv(old_csv_path, encoding='utf-8-sig')

# 重命名列
new_names = {
    'MONATSZAHL': 'Category',
    'AUSPRAEGUNG': 'AccidentType',
    'JAHR': 'Year',
    'MONAT': 'Month',
    'WERT': 'Value'
}
df = df.rename(columns=new_names)

# 删除不需要的列
cols_to_drop = ['VORJAHRESWERT', 'VERAEND_VORMONAT_PROZENT', 'ZWOELF_MONATE_MITTELWERT']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# 只保留 AccidentType == 'insgesamt'
df = df[df['AccidentType'] == 'insgesamt']

# 去掉 Month 列中为 'Summe' 的行
df = df[df['Month'] != 'Summe']

# 将 Month 转为字符串并提取最后两位作为月份
df['Month'] = df['Month'].astype(str)
df['Extracted_Month'] = df['Month'].apply(lambda x: x[-2:])
df['Year'] = df['Year'].astype(int)
df['Extracted_Month'] = df['Extracted_Month'].astype(int)

# 将 Extracted_Month 转换成两位数字符串的 Month 列
df['Month'] = df['Extracted_Month'].astype(str).str.zfill(2)

# 创建 Date 列
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%m')
df = df.drop(columns=['Extracted_Month'])

# 使用 pivot_table 将 Category 展开为列
df_overall = df.pivot_table(
    index=["Year", "Month", "Date"],
    columns="Category",
    values="Value",
    aggfunc="sum"
).reset_index()

df_overall.columns.name = None

# 设置 Date 为索引并排序
df_overall['Date'] = pd.to_datetime(df_overall['Date'])
df_overall.set_index('Date', inplace=True)
df_overall = df_overall.sort_index()

# 删除2021年及以后的数据(保留 Year <= 2020)
df_overall = df_overall[df_overall['Year'] <= 2020]

# 删除所有三种事故数量都为0的行
df_overall = df_overall[~((df_overall['Alkoholunfälle'] == 0) & 
                          (df_overall['Fluchtunfälle'] == 0) & 
                          (df_overall['Verkehrsunfälle'] == 0))]

# 将处理后的数据写入新的CSV文件
df_overall.to_csv(new_csv_path, encoding='utf-8-sig')

print(f"数据转换完成！已生成新的文件：{new_csv_path}")
print("原始文件仍在：", old_csv_path)
