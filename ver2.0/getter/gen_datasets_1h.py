import pandas as pd
import filters


base_year = 2001
count = 21
res = None 
df2 = pd.DataFrame()

save_dir = r'C:\Users\steel\dev\AI\datasets\exchange_rate\custom_1h/'

year = base_year
dir_path = r'C:\Users\steel\dev\AI\datasets\exchange_rate\1h/'

for _ in range(0, count):
    year = int(year)
    year += 1
    print(year)
    year = str(year)

    file_name = f"DAT_MT_USDJPY_1H_{year}.csv"
    path = dir_path + file_name

    df = pd.read_csv(path, index_col=["date", "time"])

    df2 = pd.concat([df2, df])


df = filters.classify(df2)

print(df)

df = filters.without_holidays(df, year=base_year, count=count)

year = base_year
for _ in range(0, count):
    year = int(year)
    year += 1
    year = str(year)
    file_name = f'DAT_MT_USDJPY_1H_{year}.csv'
    save_path = save_dir + file_name
    save_path = save_dir + file_name

    df3 = pd.DataFrame()
    for i, datetime in enumerate(df.index):
        if year in str(datetime[0]):
            df3 = pd.concat([df3, df[i:i+1]])
            print(df[i:i+1])
    print(df3)

    df3.to_csv(save_path)

