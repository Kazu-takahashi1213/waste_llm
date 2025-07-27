import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paderbornのゴミ分別情報ページのURL
URL = "https://www.asp-paderborn.de/abfall-abc/"
# ベクトルストアの保存先
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_store(url):
    """
    指定されたURLから情報を取得し、ベクトルストアを構築して保存する。
    """
    print("Webサイトからデータの読み込みを開始します...")
    # requestsでHTMLを取得し、BeautifulSoupでテキストを抽出
    try:
        response = requests.get(url)
        response.raise_for_status()  # エラーがあれば例外を発生させる
        soup = BeautifulSoup(response.content, 'html.parser')
        # mainタグなど、主要なコンテンツが含まれる部分に絞り込むとより精度が上がる
        # ここではシンプルにbody全体のテキストを取得
        text = soup.body.get_text(separator='\n', strip=True)
    except requests.RequestException as e:
        print(f"Webサイトの読み込みに失敗しました: {e}")
        return
    
    print(f"データの読み込みが完了しました。文字数: {len(text)}")

    print("テキストをチャンクに分割します...")
    # テキストを適切なサイズのチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents([text])
    print(f"分割後のチャンク数: {len(chunks)}")

    print("埋め込みモデルをロードします...")
    # 埋め込みモデルをロード
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2',
                                       model_kwargs={'device': 'cpu'})

    print("ベクトルストアを構築し、保存します...")
    # FAISSベクトルストアを構築
    db = FAISS.from_documents(chunks, embeddings)
    
    # ベクトルストアをローカルに保存
    db.save_local(DB_FAISS_PATH)
    print(f"ベクトルストアを '{DB_FAISS_PATH}' に保存しました。")

if __name__ == '__main__':
    create_vector_store(URL)
    print("知識データベースの構築が完了しました。")
