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

# 保留protein_expression_normalized原样，因为它已经是归一化的

def plot_metrics_comparison_chart(data, source_value, title):
    # 过滤特定source的数据
    filtered_data = data[data['source'] == source_value].copy()
    
    # 创建序列索引
    filtered_data['sequential_index'] = range(len(filtered_data))
    positions = list(filtered_data['sequential_index'])
    
    # 定义要绘制的列和对应的颜色
    columns_to_plot = {
        'protein_expression_normalized': 'rgb(180, 180, 180)',  # 淡灰色
        'sanofi_te_normalized': 'rgb(230, 145, 90)',  # 淡橙色
        'supra_te_normalized': 'rgb(120, 190, 150)'  # 淡绿色
    }
    
    # 创建柱状图
    fig = go.Figure()
    
    # 为每个指标添加一个trace
    for column, color in columns_to_plot.items():
        fig.add_trace(
            go.Bar(
                x=positions,
                y=filtered_data[column],
                name=column,
                marker_color=color,
                text=filtered_data[column].round(3),
                textposition='outside',
                textfont=dict(color='black')
            )
        )
    
    # 更新布局
    fig.update_layout(
        title=f'{title} - Source: {source_value} (Normalized)',
        xaxis_title='Index',
        yaxis_title='Normalized Value',
        height=600,
        width=1500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2,
        barmode='group',  # 使用分组模式显示不同指标的柱状图
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 更新坐标轴样式，确保x轴显示完整的序列索引
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        tickmode='array',
        tickvals=positions,  # 使用位置作为刻度值
        ticktext=[str(i) for i in positions],  # 将位置转换为字符串作为刻度标签
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

# 为每个source创建一个对比柱状图（使用归一化后的数据）
for source in sources:
    fig = plot_metrics_comparison_chart(
        df_normalized, 
        source, 
        'Comparison of Normalized Metrics'
    )
    # 保存图表
    fig.write_image(f'images/0225/实验数据_{source}_normalized_metrics_comparison_bar_chart.png', scale=4)
