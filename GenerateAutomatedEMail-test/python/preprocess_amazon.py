import os
import pandas as pd


def getData(path):
    q = []
    a = []
    for file in os.listdir(path):
        f = os.path.join(path, file)
        if f.endswith(".json"):
            print("Currently processing", f)
            data = open(f).readlines()
            parsed = [eval(line) for line in data]
            for x in range(len(parsed)):
                q.append(parsed[x]["question"])
                a.append(parsed[x]["answer"])
    return q, a


q, a = getData("../data/amazon")

df = {"question":q, "answer":a}
df = pd.DataFrame(df)
df.to_csv("../data/amazon.csv", index=False)
