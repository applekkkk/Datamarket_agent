import pandas as pd

df = pd.read_csv('data/6.csv')
print(df['委托数量']*df['委托价格'])