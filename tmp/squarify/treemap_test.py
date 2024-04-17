#%%
import matplotlib.pyplot as plt
import squarify
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# サンプルデータ
sizes = [25, 40, 20, 15]

# squarifyでtreemapを作成
squarify.plot(sizes=sizes, label=["Node A", "Node B", "Node C", "Node D"], alpha=0.7)

# 画像を読み込み
image_path = "/home/sumthen/2023/python_2023/tmp/scraping_postcard/image_postcard/pc000196.jpg"
original_img = Image.open(image_path)

# matplotlibの座標系を取得
ax = plt.gca()

# 各ノードに画像を挿入（サイズに合わせてリサイズ、トリミング）
for i, rectangle in enumerate(ax.patches):
    # ノードの縦幅と横幅を取得
    node_width = rectangle.get_width()
    node_height = rectangle.get_height()

    # 画像をトリミングしてリサイズ
    img_width, img_height = min(original_img.width, node_width), min(original_img.height, node_height)
    left = (original_img.width - img_width) / 2
    upper = (original_img.height - img_height) / 2
    right = left + img_width
    lower = upper + img_height

    cropped_img = original_img.crop((left, upper, right, lower)).resize((int(node_width), int(node_height)))

    # 画像を挿入
    imagebox = OffsetImage(cropped_img, zoom=1.0, resample=True)
    x_center = rectangle.get_x() + node_width / 2
    y_center = rectangle.get_y() + node_height / 2
    ab = AnnotationBbox(imagebox, (x_center, y_center), frameon=False, pad=0)
    ax.add_artist(ab)

plt.axis('off')
plt.show()

#%%
import matplotlib.pyplot as plt
import squarify
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# サンプルデータ
sizes = [25, 40, 20, 15]

# squarifyでtreemapを作成
squarify.plot(sizes=sizes, label=["Node A", "Node B", "Node C", "Node D"], alpha=0.7)

# 画像を読み込み
image_path = "/home/sumthen/2023/python_2023/tmp/scraping_postcard/image_postcard/pc000196.jpg"
original_img = Image.open(image_path)

# matplotlibの座標系を取得
ax = plt.gca()

# 各ノードに画像を挿入（ノードの幅いっぱいにリサイズ）
for i, rectangle in enumerate(ax.patches):
    # ノードの幅を取得
    node_width = rectangle.get_width()

    # 画像をノードの幅いっぱいにリサイズ
    img_height = original_img.height * (node_width / original_img.width)
    resized_img = original_img.resize((int(node_width), int(img_height)))

    # 画像を挿入
    imagebox = OffsetImage(resized_img, zoom=1.0, resample=True)
    x_center = rectangle.get_x() + node_width / 2
    y_center = rectangle.get_y() + rectangle.get_height() / 2
    ab = AnnotationBbox(imagebox, (x_center, y_center), frameon=False, pad=0)
    ax.add_artist(ab)

plt.axis('off')
plt.show()

# %%
