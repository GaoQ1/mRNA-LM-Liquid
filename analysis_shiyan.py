import pandas as pd
from scipy import stats

# df_604 = pd.read_csv("data/实验数据/604.csv")
# df_auro = pd.read_csv("data/实验数据/auro_rsv.csv")
# df_zheda_homo = pd.read_csv("data/实验数据/zheda_homo.csv")
# df_zheda_mouse = pd.read_csv("data/实验数据/zheda_mouse.csv")
df_shiyan = pd.read_csv("data/实验数据/shiyandata.csv")



import code; code.interact(local=dict(globals(), **locals()))


# pcc, _ = stats.pearsonr(df_604["predict_te"], df_604["protein_expression"])
# spcc, _ = stats.spearmanr(df_604["predict_te"], df_604["protein_expression"])


# pcc, _ = stats.pearsonr(df_auro["te"], df_auro["OD(450nm)"])
# spcc, _ = stats.spearmanr(df_auro["te"], df_auro["OD(450nm)"])


# pcc, _ = stats.pearsonr(df_zheda_homo["te"], df_zheda_homo["24h"])
# spcc, _ = stats.spearmanr(df_zheda_homo["te"], df_zheda_homo["24h"])


pcc, _ = stats.pearsonr(df_zheda_mouse["te"], df_zheda_mouse["24h"])
spcc, _ = stats.spearmanr(df_zheda_mouse["te"], df_zheda_mouse["24h"])


print(f"皮尔逊相关系数 (PCC): {pcc:.4f}")
print(f"斯皮尔曼相关系数 (SPCC): {spcc:.4f}")




