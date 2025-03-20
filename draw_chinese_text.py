from PIL import Image, ImageFont, ImageDraw
import numpy as np


def put_chinese_text(img, text):
    fontpath = 'NotoSansTC-VariableFont_wght.ttf'          # 設定字型路徑
    font = ImageFont.truetype(fontpath, 16)      # 設定字型與文字大小
    pillow_img = Image.fromarray(img)                # 將 img 轉換成 PIL 影像
    draw = ImageDraw.Draw(pillow_img)                # 準備開始畫畫
    draw.text(
        (0, 0), text,
        fill=(255, 255, 255),
        font=font
    )

    return np.array(pillow_img)
