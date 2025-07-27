import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

class WasteDisposalGuide:
    def __init__(self, csv_path):
        """
        コンストラクタ。ゴミ分別情報のCSVファイルを読み込み、ラベルをベクトル化する。
        """
        try:
            self.df = pd.read_csv(csv_path)
            # モデルをロード
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            # CSVのラベルをエンコード
            self.corpus_embeddings = self.model.encode(self.df['clip_label'].tolist(), convert_to_tensor=True)
        except FileNotFoundError:
            self.df = None
            print(f"エラー: {csv_path} が見つかりません。")
        except Exception as e:
            self.df = None
            print(f"初期化中にエラーが発生しました: {e}")

    def get_disposal_info(self, waste_name):
        """
        指定されたゴミの名称に最も意味的に近い分別情報を取得する。

        Args:
            waste_name (str): ゴミの名称。

        Returns:
            str: ゴミの分別情報。見つからない場合はその旨を返す。
        """
        if self.df is None:
            return "ゴミ分別情報が読み込まれていません。"

        # クエリ（画像認識のラベル）をエンコード
        query_embedding = self.model.encode(waste_name, convert_to_tensor=True)

        # クエリとコーパス間のコサイン類似度を計算
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # 最もスコアの高いペアを取得
        top_results = torch.topk(cos_scores, k=1)

        # 最も類似度の高いインデックスを取得
        best_match_index = top_results[1].item()
        best_match_score = top_results[0].item()

        # 類似度が一定の閾値以上の場合のみ結果を返す（例: 0.4）
        if best_match_score > 0.4:
            info = self.df.iloc[best_match_index]
            disposal = info['paderborn_disposal']
            notes = info['notes']
            matched_label = info['clip_label']

            result_text = f"「{matched_label}」として認識しました。\n\n**捨て方:** {disposal}"

            if pd.notna(notes) and notes.strip():
                result_text += f"\n\n**補足:** {notes}"
            return result_text
        else:
            return f"'{waste_name}' に関連する分別情報は見つかりませんでした。"

if __name__ == '__main__':
    guide = WasteDisposalGuide('gurbage.csv')
    if guide.df is not None:
        test_waste = "bottle"
        info = guide.get_disposal_info(test_waste)
        print(f"'{test_waste}' の捨て方: {info}")

        test_waste_unknown = "unknown item"
        info_unknown = guide.get_disposal_info(test_waste_unknown)
        print(f"'{test_waste_unknown}' の捨て方: {info_unknown}")