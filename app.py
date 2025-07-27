import streamlit as st
from PIL import Image
import os

# 他のモジュールから関数をインポート
from vision import classify_image
from rag import get_qa_chain

# Streamlitアプリのタイトルと説明
st.set_page_config(page_title="Paderborn ゴミ分別アシスタント", layout="wide")
st.title("AI ゴミ分別アシスタント for Paderborn")
st.markdown("ゴミの画像をアップロードすると、AIがその種類を判別し、Paderborn市の公式情報に基づいて捨て方を回答します。")

# QAチェーンをキャッシュしてロード
@st.cache_resource
def load_qa_chain():
    chain = get_qa_chain()
    if chain is None:
        st.error("アプリケーションの初期化に失敗しました。ベクトルストアが構築されているか確認してください。")
        st.stop()
    return chain

qa_chain = load_qa_chain()

# 画像アップロード機能
uploaded_file = st.file_uploader("ここに画像をドラッグ＆ドロップするか、ファイルを選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画面を2分割
    col1, col2 = st.columns(2)

    with col1:
        # アップロードされた画像を表示
        image = Image.open(uploaded_file)
        st.image(image, caption='アップロードされた画像', use_container_width=True)

    with col2:
        with st.spinner("画像を分析し、捨て方を調べています..."):
            # 一時ファイルとして保存
            temp_dir = "temp_images"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 1. 画像を分類
            image_label = classify_image(temp_file_path)
            st.write(f"**画像認識の結果:** `{image_label}`")

            # 2. 分類ラベルを元に質問を作成
            question = f"{image_label} の捨て方を教えてください。"
            st.write(f"**AIへの質問:** `{question}`")

            # 3. RAG+LLMチェーンで回答を生成
            try:
                result = qa_chain.invoke({"query": question})
                answer = result.get("result", "回答を生成できませんでした。")
                source_documents = result.get("source_documents")

                st.subheader("AIからの回答")
                st.info(answer)

                # 回答の根拠となったソースドキュメントを表示
                if source_documents:
                    with st.expander("回答の根拠となった情報源を見る"):
                        for doc in source_documents:
                            st.markdown(f"---")
                            st.write(doc.page_content)
            except Exception as e:
                st.error(f"回答の生成中にエラーが発生しました: {e}")

            # 一時ファイルを削除
            os.remove(temp_file_path)
