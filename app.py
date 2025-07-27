import streamlit as st
from PIL import Image
import os

# 他のモジュールから関数をインポート
from vision import classify_image
from rag import WasteDisposalGuide

# Streamlitアプリのタイトル
st.title("Paderborn ゴミ分別アプリ")

# RAGガイドのインスタンスを作成
# gurbage.csvの絶対パスを取得
csv_path = os.path.join(os.path.dirname(__file__), 'gurbage.csv')
guide = WasteDisposalGuide(csv_path)

# 画像アップロード機能
uploaded_file = st.file_uploader("ゴミの画像をアップロードしてください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # アップロードされた画像を表示
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_container_width=True)
    st.write("")
    st.write("分類中...")

    # 一時ファイルとして保存
    # classify_imageはファイルパスを引数にとるため
    temp_file_path = os.path.join("temp_images", uploaded_file.name)
    os.makedirs("temp_images", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 画像を分類
    label = classify_image(temp_file_path)
    st.write(f"画像認識の結果: **{label}**")

    # 分類ラベルを使ってゴミの捨て方を検索
    if guide.df is not None:
        disposal_info = guide.get_disposal_info(label)
        st.write("### 捨て方:")
        st.info(disposal_info)
    else:
        st.error("ゴミ分別ガイドが読み込めませんでした。")

    # 一時ファイルを削除
    os.remove(temp_file_path)
