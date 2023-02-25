import pandas as pd
import datetime
import numpy as np



def without_holidays(df, year, count):
    df2 = pd.DataFrame()
    for _ in range(0, count):
        year = int(year)
        year += 1
        year = str(year)

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
                        if date.weekday() <= 4:
                            date = year + "." + month + "." + day
                            date_data = df[date:date]
                            df2 = pd.concat([df2, date_data])
                        else:
                            print(date)
                    except:
                        print(f"{year}.{month}.{day}はデータが存在しません。")
    return df2 

def average_real(df):
    ls = []
    for i, _ in enumerate(df.index):
        open = df['open'][i]
        close = df['close'][i]
        real = abs(open - close)
        ls.append(real)
    ls = np.array(ls)
    # real = np.mean(ls)
    real = np.median(ls)
    return real

def assign(df, border=1.6, small_border=0.004, big_border=4.6):
    ls = []
    av_real = average_real(df)

    for i, item in enumerate(df.index):
        # パラメーター
        isBig = False
        isSmall = False
        isBald = False
        isHair = False
        isBeard = False
        isHammer = False
        isUmbrella = False
        isCross = False
        isDragonfly = False
        isTower = False
        isUpTrend = False
        # above == high or under == lowの場合にTrueにする
        isSkin = False
        isPositive = False

        date = item[0]
        time = item[1]
        open = df['open'][i]
        high = df['high'][i]
        low = df['low'][i]
        close = df['close'][i]
        # 陽線の時
        isPositive = True
        above = close
        under = open
        name = "陽"
        # 陰線の時
        if open > close:
            isPositive = False
            above = open
            under = close
            name = "陰"
        
        # 全く同じだとZeroDevisionErrorが生じるので微調整
        if above == high:
            high += 0.001
            isSkin = True
        if under == low:
            low += 0.001
            isSkin= True

        real = above - under
        # 大線
        if real >= av_real * big_border:
            isBig = True
        # 坊主
        if (high-above<=small_border) & (under-low<=small_border) & (above-under>=0.05):
            isBald = True
        # 上影線
        elif high-above>=(under-low)*border:
            isHair = True
            # トンカチ
            if (under-low<=small_border) & (high-above>real):
                isHammer = True
            # トウバ
            if (above-under)<=small_border:
                isTower = True
        # 下影線
        elif under-low>=(high-above)*border:
            isBeard = True
            # カラカサ
            if (high-above<=small_border) & (under-low>real):
                isUmbrella = True
            # トンボ
            if (above-under)<=small_border:
                isDragonfly = True
        # 小線の場合
        else:
            isSmall = True
            if above-under<=small_border:
                isCross = True
        if isSkin:
            open = round(open, 3)
            high = round(high, 3)
            low = round(low, 3)
            close = round(close, 3)
        sma5 = round(df['sma5'][i], 3)
        sma25 = round(df['sma25'][i], 3)
        sma50 = round(df['sma50'][i], 3)
        sma100 = round(df['sma100'][i], 3)
        data = [date, time, open, high, low, close, isPositive, isBig, isSmall, isBald, isHair, isBeard, isHammer, isUmbrella, isCross, isDragonfly, isTower, isUpTrend, sma5, sma25, sma50, sma100]
        print(data)
        ls.append(data)
    df = pd.DataFrame(ls, columns=['date', 'time', 'open', 'high', 'low', 'close', 'isPositive', 'isBig', 'isSmall', 'isBald', 'isHair', 'isBeard', 'isHammer', 'isUmbrella', 'isCross', 'isDragonfly', 'isTower', 'isUpTrend', 'sma5', 'sma25', 'sma50', 'sma100'])
    df = df.set_index(['date', 'time'])

    return df


def classify(df):
    x = [item[1] for item in df.index]
    df['sma5'] = df['close'].rolling(5).mean().shift()
    df['sma25'] = df['close'].rolling(25).mean().shift()
    df['sma50'] = df['close'].rolling(50).mean().shift()
    df['sma100'] = df['close'].rolling(100).mean().shift()
    df = assign(df)
    if df.__len__() == 0:
        return None
    return df
    


