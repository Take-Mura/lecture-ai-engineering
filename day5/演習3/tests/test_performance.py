import os
import sys
import pytest
from sklearn.model_selection import train_test_split

# main.py が置かれているディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import DataLoader, ModelTester  # adjust import path if necessary


@pytest.fixture(scope="module")
def titanic_data():
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_inference_time(benchmark, titanic_data):
    """pytest-benchmark で推論時間を計測"""
    X_train, X_test, y_train, y_test = titanic_data
    model = ModelTester.train_model(X_train, y_train)

    # 推論部分のみをベンチマーク
    benchmark(model.predict, X_test) 
