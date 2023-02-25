import datetime
import pandas as pd


def filter_holidays():
    

months = list(range(1, 13))
days = list(range(1, 31))

for month in months:
        month = str(month)
        if len(month) == 1:
            month = "0" + month
        for day in days:
            day = str(day)
            if len(day) == 1:
                day = "0" + day
            try:
                date = datetime.date(year=int(year), month=int(month), day=int(day))
                # 5以上の時、土日
                if date.weekday() <= 6:
                    date = year + "." + month + "." + day
                    date_data = df[date:date]
                    # df2 = pd.concat([df2, date_data])
                    res = classify.classify(date_data)
                    
            except:
                print(f"{year}.{month}.{day}はデータが存在しません。") 