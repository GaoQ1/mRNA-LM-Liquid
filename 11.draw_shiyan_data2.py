import pandas as pd
import plotly.graph_objs as go
import numpy as np

# 读取数据
data_example = "shiyandata_with_te.csv"
df_data_example = pd.read_csv(data_example)

# 对每个source的supra_te和sanofi_te进行最大最小值归一化
df_normalized = df_data_example.copy()

# 定义source值
sources = ['604', 'auro_rsv', 'zheda_homo', 'zheda_mouse']

# 对每个source分别进行归一化处理
for source in sources:
    source_mask = df_normalized['source'] == source
    
    # 对supra_te进行归一化
    min_supra = df_normalized.loc[source_mask, 'supra_te'].min()
    max_supra = df_normalized.loc[source_mask, 'supra_te'].max()
    if max_supra > min_supra:  # 避免除以零
        df_normalized.loc[source_mask, 'supra_te_normalized'] = (df_normalized.loc[source_mask, 'supra_te'] - min_supra) / (max_supra - min_supra)
    else:
        df_normalized.loc[source_mask, 'supra_te_normalized'] = 0
    
    # 对sanofi_te进行归一化
    min_sanofi = df_normalized.loc[source_mask, 'sanofi_te'].min()
    max_sanofi = df_normalized.loc[source_mask, 'sanofi_te'].max()
    if max_sanofi > min_sanofi:  # 避免除以零
        df_normalized.loc[source_mask, 'sanofi_te_normalized'] = (df_normalized.loc[source_mask, 'sanofi_te'] - min_sanofi) / (max_sanofi - min_sanofi)
    else:
        df_normalized.loc[source_mask, 'sanofi_te_normalized'] = 0

# 计算差值
df_normalized['supra_te_diff'] = df_normalized['supra_te_normalized'] - df_normalized['protein_expression_normalized']
df_normalized['sanofi_te_diff'] = df_normalized['sanofi_te_normalized'] - df_normalized['protein_expression_normalized']

def plot_metrics_difference_chart(data, source_value, title):
    # 过滤特定source的数据
    filtered_data = data[data['source'] == source_value].copy()
    
    # 创建序列索引
    filtered_data['sequential_index'] = range(len(filtered_data))
    positions = list(filtered_data['sequential_index'])
    
    # 定义要绘制的列和对应的颜色
    columns_to_plot = {
        'sanofi_te_diff': 'rgb(230, 145, 90)',   # 淡橙色
        'supra_te_diff': 'rgb(120, 190, 150)'    # 淡绿色
    }
    
    # 创建柱状图
    fig = go.Figure()
    
    # 为每个差值添加一个trace
    for column, color in columns_to_plot.items():
        display_name = 'Supra TE - Protein Expression' if column == 'supra_te_diff' else 'Sanofi TE - Protein Expression'
        fig.add_trace(
            go.Bar(
                x=positions,
                y=filtered_data[column],
                name=display_name,
                marker_color=color,
                text=filtered_data[column].round(3),
                textposition='outside',
                textfont=dict(color='black')
            )
        )
    
    # 更新布局
    fig.update_layout(
        title=f'{title} - Source: {source_value} (Normalized Differences)',
        xaxis_title='Index',
        yaxis_title='Difference in Normalized Values',
        height=600,
        width=1500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # 更新坐标轴样式
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        tickmode='array',
        tickvals=positions,
        ticktext=[str(i) for i in positions],
        tickangle=0
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    return fig

# 为每个source创建一个差值对比柱状图
for source in sources:
    fig = plot_metrics_difference_chart(
        df_normalized, 
        source, 
        'Comparison of Normalized Metrics Differences'
    )
    # 保存图表
    fig.write_image(f'images/0226/实验数据_{source}_normalized_metrics_differences_bar_chart.png', scale=4)
