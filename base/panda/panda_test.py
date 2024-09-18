import pandas as pd


def test1():
    data = [['Google',10],['Runoob',12],['Wiki',13]]
    df = pd.DataFrame(data,columns=['Site','Age'],dtype=float)
    print(df)


def test2():
    data = {'Site': ['Google', 'Runoob', 'Wiki'], 'Age': [10, 12, 13]}
    df = pd.DataFrame(data)
    print(df)


def test3():
    df = pd.DataFrame(columns=['Site', 'Age'], dtype=float)
    df = df.append({'Site': 'Google', 'Age': 10}, ignore_index=True)
    print(df)


def test4():
    df = pd.DataFrame(columns=['Site', 'Age'], dtype=float)
    df = df.append(pd.Series(['Google', 10], index=df.columns), ignore_index=True)
    print(df)


def test5():
    df = pd.DataFrame(columns=['Site', 'Age'], dtype=float)
    df = df.append(pd.Series(['Google', 10], index=df.columns), ignore_index=True)
    df = df.append(pd.Series(['Apple', 12], index=df.columns), ignore_index=True)
    print(df.sample(1))


if __name__ == "__main__":
    test5()
