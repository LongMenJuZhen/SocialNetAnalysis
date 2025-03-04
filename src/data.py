import pandas as pd

""" 董监高 """
# 有些行会多一列，导致读取失败，这里直接跳过这些行
# df = pd.read_csv('data/raw/董监高/TMT_FIGUREINFO.csv', on_bad_lines='skip')
# df.to_csv('data/processed/董监高/TMT_FIGUREINFO_fix_bad_line.csv', index=False)
# df = pd.read_csv('data/processed/董监高/TMT_FIGUREINFO_fix_bad_line.csv')

# 保存Reptdt符合2010-07-20格式的行到一个新的csv文件，不符合的行见invalid_reptdt.csv
# df['Reptdt'] = df['Reptdt'].astype(str)
# invalid_reptdt = df[df['Reptdt'].str.match(r'\d{4}-\d{2}-\d{2}')]
# invalid_reptdt.to_csv('data/processed/董监高/TMT_FIGUREINFO_valid_reptdt.csv', index=False)

# 合并同一个人（即同一个PersonID）的记录，对于每一个属性，根据Reptdt（形如2010-07-20）取最新的不为空的记录作为结果。
# df = pd.read_csv('data/processed/董监高/TMT_FIGUREINFO_valid_reptdt.csv')
# df['Reptdt'] = pd.to_datetime(df['Reptdt'])
# df = df.sort_values(by='Reptdt')
# df = df.groupby('PersonID').apply(lambda x: x.ffill().iloc[-1]).reset_index(drop=True)
# df.to_csv('data/processed/董监高/TMT_FIGUREINFO_unique_person.csv', index=False)

# 只保留关于董监高本人的属性，即PersonID，Name，Nationality，NativePlace，BirthPlace，Gender，Age，University，Degree，Major，Profession，Resume
# df = pd.read_csv('data/processed/董监高/TMT_FIGUREINFO_unique_person.csv')
# df = df[['PersonID','Name','Nationality','NativePlace','BirthPlace','Gender','Age','University','Degree','Major','Profession','Resume','Funback','OveseaBack','Academic','FinBack']]
# df.to_csv('data/processed/董监高/TMT_FIGUREINFO_final.csv', index=False)


""" 职位 """
# 对于相同PersonID和Stkcd的记录只保留一行，对于每一个属性，根据Reptdt（形如2010-07-20）取最新的不为空的记录作为结果。
# df = pd.read_csv('data/processed/董监高/TMT_FIGUREINFO_valid_reptdt.csv')
# df['Reptdt'] = pd.to_datetime(df['Reptdt'])
# df = df.sort_values(by='Reptdt')
# df = df.groupby(['PersonID','Stkcd']).apply(lambda x: x.ffill().iloc[-1]).reset_index(drop=True)
# df = df[['PersonID','Name','Nationality','NativePlace','BirthPlace','Gender','Age','University','Degree','Major','Profession','Resume']]
# df.to_csv('data/processed/职位/TMT_FIGUREINFO_unique_both.csv', index=False)

df = pd.read_csv('data/raw/董监高/TMT_POSITION.csv')
# # 找到相同PersonID和Stkcd出现最多的组合
# most_common_combination = df.groupby(['PersonID', 'Stkcd']).size().idxmax()
# # 过滤出这种组合的记录
# filtered_df = df[(df['PersonID'] == most_common_combination[0]) & (df['Stkcd'] == most_common_combination[1])]
# # 保存这种组合的记录
# filtered_df.to_csv('data/processed/职位/TMT_POSITION_most_common_combination.csv', index=False)


# 对于相同PersonID，PositionID和Stkcd的记录只保留一行，对于每一个属性，根据Reptdt（形如2010-07-20）取最新的不为空的记录作为结果。
df['Reptdt'] = pd.to_datetime(df['Reptdt'])
df = df.sort_values(by='Reptdt')
df = df.groupby(['PersonID','PositionID','Stkcd']).apply(lambda x: x.ffill().iloc[-1]).reset_index(drop=True)
df.to_csv('data/processed/职位/TMT_POSITION_unique_both.csv', index=False)