import pandas as pd

path = r"C:\Users\steel\dev\AI\datasets\exchange_rate\custom_1h\DAT_MT_USDJPY_1H_2001.csv"
df = pd.read_csv(path)
print(df)
