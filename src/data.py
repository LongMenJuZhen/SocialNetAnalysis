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
# df.to_csv('data/processed/职位/TMT_FIGUREINFO_unique_both.csv')

# df = pd.read_csv('data/raw/董监高/TMT_POSITION.csv')
# # 找到相同PersonID和Stkcd出现最多的组合
# most_common_combination = df.groupby(['PersonID', 'Stkcd']).size().idxmax()
# # 过滤出这种组合的记录
# filtered_df = df[(df['PersonID'] == most_common_combination[0]) & (df['Stkcd'] == most_common_combination[1])]
# # 保存这种组合的记录
# filtered_df.to_csv('data/processed/职位/TMT_POSITION_most_common_combination.csv', index=False)


# 对于相同PersonID，PositionID和Stkcd的记录只保留一行，对于每一个属性，根据c（形如2010-07-20）取最新的不为空的记录作为结果。
# df['Reptdt'] = pd.to_datetime(df['Reptdt'])
# df = df.sort_values(by='Reptdt')
# df = df.groupby(['PersonID','PositionID','Stkcd']).apply(lambda x: x.ffill().iloc[-1]).reset_index(drop=True)
# df.to_csv('data/processed/职位/TMT_POSITION_unique_both.csv', index=False)

# 根据PersonID和Stkcd查询TMT_FIGUREINFO_unique_both.csv，把其独有的属性添加到TMT_POSITION_unique_both.csv里作为TMT_POSITION_final.csv
# df1 = pd.read_csv('data/processed/职位/TMT_FIGUREINFO_unique_both.csv')
# df2 = pd.read_csv('data/processed/职位/TMT_POSITION_unique_both.csv')
# df = pd.merge(df2, df1, on=['PersonID','Stkcd','PositionID'], how='left')
# df.to_csv('data/processed/职位/TMT_POSITION_merge.csv', index=False)

# 去掉TMT_POSITION_merge.csv里的Reptdt_y，Name_y，Nationality，NativePlace，NatAreaCode，BirthPlace，BirAreaCode，Gender，Age，University，Degree，Major，Profession，Resume，Position_y，ServicePositionID，Funback，OveseaBack，Academic，FinBack，Director_TotCO，Director_ListCO，Stkcd_director这些属性
# df = pd.read_csv('data/processed/职位/TMT_POSITION_merge.csv')
# df = df.drop(['Reptdt_y','Name_y','Nationality','NativePlace','NatAreaCode','BirthPlace','BirAreaCode','Gender','Age',
#               'University','Degree','Major','Profession','Resume','Position_y','ServicePositionID','Funback',
#               'OveseaBack','Academic','FinBack','Director_TotCO','Director_ListCO','Stkcd_director'], axis=1)
# df = df.rename(columns={'Reptdt_x':'Reptdt','Name_x':'Name','Position_x':'Position'})
# df.to_csv('data/processed/职位/TMT_POSITION_final.csv', index=False)

""" 企业 """
# # 忽略第二行和第三行，读取TMT_POSITION_final.csv
# df = pd.read_csv('data/raw/FI_T7.csv', skiprows=[1, 2])
# # 根据列筛选行: 'Accper'
# df = df[df['Accper'].str.contains("2023", regex=False, na=False, case=False)]
# # 根据列筛选行: 'Typrep'
# df = df[df['Typrep'].str.contains("A", regex=False, na=False, case=False)]
# # 把F070101B绝对值大于 512，F070201B绝对值大于128，F070301B绝对值大于 1024的行删除
# df = df[(df['F070301B'] >= -50) & (df['F070301B'] <= 200)]

# # 把有相同的Stkcd的记录合并在一起，对于每一个属性，对于F070101B，F070201B，F070301B，取其标准差，其余属性根据Accper（形如2010-07-20）取最新的不为空的记录作为结果。
# df['Accper'] = pd.to_datetime(df['Accper'])
# df = df.sort_values(by='Accper')
# df = df.groupby('Stkcd').agg({'Stkcd':'first','Accper':'max','F070301B':'median'})
# df.to_csv('data/processed/企业/FI_T7_final.csv', index=False) 
""" 不连锁的董监高 """
# import pandas as pd

# # 读取两个 CSV 文件
# df1 = pd.read_csv('data/processed/董监高/TMT_FIGUREINFO_final.csv')
# df2 = pd.read_csv('data/processed/职位/TMT_POSITION_final.csv')

# # 提取两个 DataFrame 中的 PersonID 列
# person_ids_1 = set(df1['PersonID'])
# person_ids_2 = set(df2['PersonID'])

# # 找出 df1 中独有的 PersonID
# unique_to_df1 = person_ids_1 - person_ids_2

# # 找出 df2 中独有的 PersonID
# unique_to_df2 = person_ids_2 - person_ids_1

# # 将结果转换为 DataFrame 并保存到 CSV 文件
# unique_to_df1_df = pd.DataFrame(list(unique_to_df1), columns=['PersonID'])
# unique_to_df2_df = pd.DataFrame(list(unique_to_df2), columns=['PersonID'])

# unique_to_df1_df.to_csv('data/processed/董监高/unique_person_ids_in_TMT_FIGUREINFO_final.csv', index=False)
# unique_to_df2_df.to_csv('data/processed/职位/unique_person_ids_in_TMT_POSITION_final.csv', index=False)

# print("独有的 PersonID 已保存到相应的 CSV 文件中。")
# import pandas as pd

# # 读取两个 CSV 文件
# df1 = pd.read_csv('data/processed/董监高/TMT_FIGUREINFO_final.csv')
# df2 = pd.read_csv('data/processed/职位/TMT_POSITION_final.csv')

# # 提取两个 DataFrame 中的 PersonID 列
# person_ids_1 = set(df1['PersonID'])
# person_ids_2 = set(df2['PersonID'])

# # 找出 df1 中存在但在 df2 中不存在的 PersonID
# unique_to_df1 = person_ids_1 - person_ids_2

# # 读取 TMT_FIGUREINFO_fix_bad_line.csv 文件
# df_fix_bad_line = pd.read_csv('data/processed/董监高/TMT_FIGUREINFO_fix_bad_line.csv')

# # 找出 TMT_FIGUREINFO_fix_bad_line.csv 中关于这些独有 PersonID 的记录
# unique_records = df_fix_bad_line[df_fix_bad_line['PersonID'].isin(unique_to_df1)]

# # 保存结果到新的 CSV 文件
# unique_records.to_csv('data/processed/董监高/unique_person_records_in_TMT_FIGUREINFO_fix_bad_line.csv', index=False)

# print("独有的 PersonID 记录已保存到新的 CSV 文件中。")