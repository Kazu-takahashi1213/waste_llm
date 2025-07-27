from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from llm_client import get_llm_chain

DB_FAISS_PATH = 'vectorstore/db_faiss'

def get_qa_chain():
    """
    ベクトルストアとLLMChainをロードし、QAチェーンを構築する。
    """
    # 埋め込みモデルをロード
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2',
                                       model_kwargs={'device': 'cpu'})
    
    # ベクトルストアをロード
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"ベクトルストアのロード中にエラーが発生しました: {e}")
        print("`build_vector_store.py` を実行して、ベクトルストアを構築してください。")
        return None

    # LLMChainを取得
    llm_chain = get_llm_chain()
    if llm_chain is None:
        return None

    # QAチェーンを構築
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_chain.llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': llm_chain.prompt}
    )
    
    return qa_chain

if __name__ == '__main__':
    # このファイルが直接実行された場合のテストコード
    print("RAGシステムのテスト（直接実行）")
    # qa_chain = get_qa_chain()
    # if qa_chain:
    #     # テストクエリ
    #     query = "Wie entsorge ich eine Plastikflasche?"
    #     result = qa_chain.invoke({"query": query})
    #     print("クエリ:", query)
    #     print("回答:", result["result"])
    # else:
    #     print("QAチェーンの構築に失敗しました。")