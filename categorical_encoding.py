# mapping dictionary 
import pandas as pd

df = pd.read_csv("winequality-red.csv")

quality_mapping = {
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
}

df.loc[:, "quality"] = df.quality.map(quality_mapping)
