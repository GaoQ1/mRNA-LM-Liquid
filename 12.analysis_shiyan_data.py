import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy import stats

# 读取数据
data_example = "shiyandata_with_te.csv"
df_data_example = pd.read_csv(data_example)

sources = ['604', 'auro_rsv', 'zheda_homo', 'zheda_mouse']

# 计算每个source的相关系数
for source in sources:
    # 筛选特定source的数据
    source_data = df_data_example[df_data_example['source'] == source]
    
    # 计算protein_expression_normalized和supra_te的相关系数
    pcc_supra = stats.pearsonr(source_data['protein_expression_normalized'], source_data['supra_te'])[0]
    spcc_supra = stats.spearmanr(source_data['protein_expression_normalized'], source_data['supra_te'])[0]
    
    # 计算protein_expression_normalized和sanofi_te的相关系数
    pcc_sanofi = stats.pearsonr(source_data['protein_expression_normalized'], source_data['sanofi_te'])[0]
    spcc_sanofi = stats.spearmanr(source_data['protein_expression_normalized'], source_data['sanofi_te'])[0]
    
    print(f"\nSource: {source}")
    print(f"Protein Expression vs Supra TE:")
    print(f"PCC: {pcc_supra:.4f}")
    print(f"SPCC: {spcc_supra:.4f}")
    print(f"\nProtein Expression vs Sanofi TE:")
    print(f"PCC: {pcc_sanofi:.4f}")
    print(f"SPCC: {spcc_sanofi:.4f}")

# import code; code.interact(local=dict(globals(), **locals()))


