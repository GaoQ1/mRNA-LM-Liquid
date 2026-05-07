import pandas as pd
import plotly.graph_objs as go

sources = ["604", "auro_rsv", "zheda_homo", "zheda_mouse"]

# 读取并合并所有数据
dfs = []
for source in sources:
    data_path = f"result/{source}/result/{source}_with_predictions.csv"
    df = pd.read_csv(data_path)
    df['source'] = source  # 添加source列
    dfs.append(df)

df_data_example = pd.concat(dfs, ignore_index=True)

# import code; code.interact(local=dict(globals(), **locals()))

def plot_single_source_bar_chart(data, source, value_column, title):
    # 创建柱状图
    fig = go.Figure()
    
    # 筛选特定source的数据
    source_data = data[data['source'] == source].copy()
    source_data['sequential_index'] = range(len(source_data))
    
    # 为不同的source定义不同的颜色
    color_map = {
        '604': 'rgb(59, 89, 152)',      # 深蓝色
        'auro_rsv': 'rgb(211, 84, 0)',  # 橙色
        'zheda_homo': 'rgb(46, 204, 113)', # 绿色
        'zheda_mouse': 'rgb(142, 68, 173)'  # 紫色
    }
    
    # 添加柱状图
    fig.add_trace(
        go.Bar(
            name=source,
            x=source_data['sequential_index'],
            y=source_data[value_column],
            marker_color=color_map[source],
            text=source_data[value_column].round(3),
            textposition='outside',
            textfont=dict(size=10, color='black'),
            hovertemplate=f"Index: %{{x}}<br>{value_column}: %{{y:.3f}}<extra></extra>"
        )
    )
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text=f"{title} - {source}",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis_title='Sequential Index',
        yaxis_title=value_column.upper(),
        height=600,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        bargap=0.15,
        font=dict(size=12)
    )
    
    # 更新坐标轴样式
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        dtick=1  # 设置x轴刻度间隔为1
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

# 为每个source生成单独的TE柱状图
for source in sources:
    fig_te = plot_single_source_bar_chart(
        df_data_example, 
        source, 
        'te', 
        'Translation Efficiency (TE)'
    )
    fig_te.write_image(f'images/实验数据_{source}_te_bar_chart.png', scale=4)

# 为每个source生成单独的protein expression柱状图
for source in sources:
    fig_pe = plot_single_source_bar_chart(
        df_data_example, 
        source, 
        'protein_expression', 
        'Protein Expression (PE)'
    )
    fig_pe.write_image(f'images/实验数据_{source}_pe_bar_chart.png', scale=4)

