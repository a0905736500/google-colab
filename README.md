# 🚀 如何使用 Google Colab 免費算力訓練自己的模型
> **組員名單**：11224134 朱原均、11224116 賴至璸  
> **展示範例**：PyTorch MNIST 手寫數字辨識模型訓練

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Free%20GPU-F9AB00)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 專案主題簡介

本專案主要探討與實作**「如何利用 Google Colab 提供的免費雲端算力來完成深度學習模型的訓練與保存」**。

在機器學習與深度學習的開發過程中，本機硬體資源（如 GPU 顯示卡）常是初學者的主要瓶頸。Google Colab 提供免費的 T4 GPU/TPU 算力，非常適合用於模型實驗與訓練。

為了完整呈現 Colab 的教學與實作流程，本專案以經典的 **MNIST 手寫數字辨識** 為範例，展示從「雲端環境設定」到「模型訓練與資料持久化保存」的完整端到端（End-to-End）工作流程。

---

## 🎯 核心實作流程 (Colab 算力運用重點)

1. **雲端硬體加速器啟用**：學習在 Colab 中檢查並切換 CPU / CUDA GPU 算力環境。
2. **Google Drive 掛載與自動化目錄管理**：解決 Colab 免費版每次斷線會清空暫存區的問題，實作自動掛載雲端硬碟並建立專案路徑。
3. **模型訓練與雲端持久化**：訓練過程中的模型權重檔（`.pth`）、訓練檢查點（`.pt`）以及評估圖表（`.png`）皆即時自動存回 Google Drive。
4. **訓練結果可視化與評估**：使用 `matplotlib` 監控 Loss/Accuracy 學習曲線，並實作測試集圖形預測展示。

---

## 🏗️ 範例模型架構與配置

作為展示範例，我們建立了一個多層感知機（MLP）模型進行手寫數字辨識：

```text
[ Input: 1x28x28 ]
       │
   [ Flatten ]  ───> 轉為 784 維向量
       │
   [ Linear ]   ───> Output: 256
       │
     [ ReLU ]   ───> 激活函數
       │
   [ Dropout ]  ───> p=0.2 (防止過擬合)
       │
   [ Linear ]   ───> Output: 128
       │
     [ ReLU ]   ───> 激活函數
       │
   [ Linear ]   ───> Output: 10 (0~9 類別 Logits)
```
## 🎲 隨機種子設定 (Reproducibility)

為了確保模型訓練與實驗結果的可重複性（Reproducibility），我們固定了 Python 原生 `random` 與 PyTorch 的隨機種子（Seed）：

```python
import random
import torch

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("AI 手寫數字辨識模型訓練開始")
print("=" * 60)
```
### ⚡ 運算環境自動偵測 (Environment Setup)

程式會自動判斷 Google Colab 當前是否已啟用 GPU 加速：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n[1/8] 運算環境")
print("PyTorch 版本：", torch.__version__)
print("使用裝置：", device)

if torch.cuda.is_available():
    print("GPU 型號：", torch.cuda.get_device_name(0))
else:
    print("目前使用 CPU，訓練速度可能較慢。")
    print("如需 GPU，請在 Colab 設定硬體加速器。")
```
print("\n[2/8] 連接 Google Drive")

drive.mount("/content/drive")

### 建立專案資料夾
```python
PROJECT_DIR = "/content/drive/MyDrive/MNIST_AI_Project"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("專案資料夾：", PROJECT_DIR)
```
### 📦 資料集下載與批次處理 (Data Pipeline)

使用 `torchvision.datasets` 自動下載 MNIST 資料集，並利用 `DataLoader` 進行平行化的批次處理：

```python
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="/content/mnist_data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="/content/mnist_data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
```
