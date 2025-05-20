import os
import sys
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# main.py が置かれているディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import DataLoader, ModelTester  # adjust import path if necessary


@pytest.fixture(scope="module")
def titanic_data():
    # データロード＆前処理
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_accuracy_precision_recall_f1(titanic_data):
    """精度・適合率・再現率・F1の検証"""
    X_train, X_test, y_train, y_test = titanic_data

    # モデル学習
    model = ModelTester.train_model(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)

    # 各種指標算出
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # しきい値チェック（例：0.75以上）
    assert acc >= 0.75, f"Accuracy がベースラインを下回っています: {acc:.4f}"
    assert prec >= 0.75, f"Precision がベースラインを下回っています: {prec:.4f}"
    assert rec >= 0.75, f"Recall がベースラインを下回っています: {rec:.4f}"
    assert f1 >= 0.75, f"F1-score がベースラインを下回っています: {f1:.4f}"
