import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# モデルとプロセッサをロード
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

def classify_image(image_path):
    """
    指定された画像を分類し、最も可能性の高いラベルを返す。

    Args:
        image_path (str): 画像ファイルのパス。

    Returns:
        str: 分類されたラベル。
    """
    try:
        # 画像を開いてRGBに変換
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return "画像ファイルが見つかりません。"
    except Exception as e:
        return f"画像の読み込み中にエラーが発生しました: {e}"

    # 画像を前処理
    inputs = processor(images=image, return_tensors="pt")

    # モデルで推論
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 最も確率の高いクラスを取得
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class

if __name__ == '__main__':
    # テスト用の画像パス（適宜変更してください）
    test_image = "path/to/your/test_image.jpg"
    label = classify_image(test_image)
    print(f"この画像は '{label}' に分類されました。")