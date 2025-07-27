import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_llm_chain():
    """
    Hugging FaceのLLMとプロンプトを組み合わせて、LLMChainを生成する。
    Hugging FaceのアクセストークンはStreamlitのsecretsから取得する。
    """
    try:
        # StreamlitのsecretsからHugging Faceのアクセストークンを取得
        huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except FileNotFoundError:
        st.error("Hugging Faceのアクセストークンが見つかりません。Streamlitのsecretsに設定してください。")
        st.stop()
    except KeyError:
        st.error("`HUGGINGFACEHUB_API_TOKEN` がsecretsに設定されていません。")
        st.stop()

    # Hugging Faceのエンドポイントを設定
    # モデル: meta-llama/Meta-Llama-3-8B-Instruct
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=huggingfacehub_api_token,
        temperature=0.7,
        max_new_tokens=512,
        top_k=50,
        top_p=0.95,
    )

    # プロンプトテンプレートの定義
    prompt_template = """
    あなたはドイツのパーダーボルン市におけるゴミ分別の専門アシスタントです。
    以下のコンテキスト情報に基づいて、ユーザーからの質問に日本語で分かりやすく、親切に回答してください。

    コンテキスト:
    {context}

    質問:
    {question}

    回答:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # LLMChainを作成
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain

if __name__ == '__main__':
    # このファイルが直接実行された場合のテストコード
    # Streamlitのsecretsは直接実行時には利用できないため、注意
    print("LLMクライアントのテスト（直接実行）")
    # 実際の利用にはStreamlitアプリ内での実行が必要
    # chain = get_llm_chain()
    # print(chain)