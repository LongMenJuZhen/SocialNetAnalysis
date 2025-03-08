# 已从 URI 中加载变量“df”: /Users/313/git_project/SocialNetAnalysis/data/processed/董监高/TMT_FIGUREINFO_final.csv
import pandas as pd
df = pd.read_csv(r'/Users/313/git_project/SocialNetAnalysis/data/processed/董监高/TMT_FIGUREINFO_final.csv')

# Fill NaN values in 'Nationality' with the mode
df['Nationality'].fillna(df['Nationality'].mode()[0], inplace=True)

# 删除列: 'NativePlace'
df = df.drop(columns=['NativePlace'])

# 删除列: 'BirthPlace'
df = df.drop(columns=['BirthPlace'])

# 删除列: 'University'
df = df.drop(columns=['University'])

# Fill NaN values in 'Degree' with the mean and convert to integer
df['Degree'].fillna(df['Degree'].mean(), inplace=True)
df['Degree'] = df['Degree'].astype(int)

# 删除列: 'Major'
df = df.drop(columns=['Major'])

# 删除列: 'Profession'
df = df.drop(columns=['Profession'])