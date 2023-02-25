import pandas as pd
import datetime
import plot


dir_path = r"C:\Users\Tetsuya Shinagawa\dev\share\dl\data/"

base_year = 2020
count = 1
year = base_year

df2 = pd.DataFrame()

for _ in range(0, count):
    year = int(year)
    year += 1
    year = str(year)
    
    file_name = f'DAT_MT_USDJPY_1H_{year}.csv'
    path = dir_path + file_name
    df = pd.read_csv(path, header=0)
    df = df.set_index(['date', 'time'])
    for month in list(range(1, 13)):
        month = str(month)
        if len(month) == 1:
            month = "0" + month
        for day in list(range(1, 31)):
            day = str(day)
            if len(day) == 1:
                day = "0" + day
            date = year + "." + month + "." + day
            try:
                date_data = df[date:date]
                if len(date_data) != 0:
                    plot.show_fig(date_data)
            except:
                print(f"{date}は存在しません。")

