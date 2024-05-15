
# %%
import plotly.graph_objects as go

# ダミーデータ
values = [10, 15, 8, 12, 7]
labels = ['A', 'B', 'C', 'D', 'F']

# 画像のパスリスト（各ノードに対応する画像）
image_paths = ['/home/sumthen/2023/python_2023/tmp/scraping_postcard/image_postcard/pc000196.jpg',
                '/home/sumthen/2023/python_2023/tmp/scraping_postcard/image_postcard/pc000198.jpg',
                '/home/sumthen/2023/python_2023/tmp/scraping_postcard/image_postcard/pc000200.jpg',
                '/home/sumthen/2023/python_2023/tmp/scraping_postcard/image_postcard/pc000201.jpg',
                '/home/sumthen/2023/python_2023/tmp/scraping_postcard/image_postcard/pc000211.jpg']

# Treemap の作成
fig = go.Figure()

for i in range(len(labels)):
    fig.add_trace(go.Treemap(
        labels=[f"{labels[i]}<br>{values[i]}" for i in range(len(labels))],
        parents=[""] * len(labels),
        values=values,
        marker=dict(
            colors=["lightblue", "lightgreen", "lightpink", "lightyellow", "lightcoral"],
            line=dict(width=1, color="black"),
        )
    ))

# 各ノードに画像を挿入
for i, (label, image_path) in enumerate(zip(labels, image_paths)):
    fig.add_layout_image(
        source=image_path,
        x=0.2 * (i + 0.5),  # 画像のx座標を調整
        y=0.9,  # 画像のy座標を調整
        xref="paper",
        yref="paper",
        sizex=0.2,
        sizey=0.2,
        opacity=1,
        xanchor='center',
        yanchor='middle',
    )

# レイアウトの設定
fig.update_layout(
    showlegend=False,
    annotations=[
        dict(
            x=0.5,
            y=0.5,
            showarrow=False,
            text="",
        )
    ]
)

# プロットの表示
fig.show()

#%%
import plotly.graph_objects as go

# サンプルデータ
sizes = [25, 40, 20, 15]
labels = ["Node A", "Node B", "Node C", "Node D"]

# 画像のURL
image_url = "/home/sumthen/2023/python_2023/tmp/scraping_postcard/image_postcard/pc000196.jpg"

# 画像を挿入するHTML
image_html = f'<img src="{image_url}" width="100%" height="100%">'

# カスタムのHTML要素を構築
custom_data = [
    f'<div style="width:100%;height:100%;text-align:center;">{image_html}</div>'
    for _ in sizes
]

# Plotly Treemapを作成
fig = go.Figure(go.Treemap(
    labels=labels,
    parents=["", "", "", ""],
    values=sizes,
    hoverinfo="label+value",
    customdata=custom_data,
    texttemplate="%{customdata}",
))

# レイアウトの調整
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

# 表示
fig.show()


# %%
