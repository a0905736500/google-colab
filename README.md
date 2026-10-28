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
<img width="577" height="103" alt="image" src="https://github.com/user-attachments/assets/036dad40-729e-4a54-8b92-72ba28fed00e" />

### 📁 建立專案資料夾
```python
PROJECT_DIR = "/content/drive/MyDrive/MNIST_AI_Project"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("專案資料夾：", PROJECT_DIR)
```
<img width="1026" height="93" alt="image" src="https://github.com/user-attachments/assets/b223c7c1-841a-4555-8fdb-594e567a1fd3" />

### 📦 資料集下載與批次處理 (Data Pipeline)

使用 `torchvision.datasets` 自動下載 MNIST 資料集，並利用 `DataLoader` 進行平行化的批次處理：

```python
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="/content/mnist_data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="/content/mnist_data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
```
<img width="623" height="107" alt="image" src="https://github.com/user-attachments/assets/241df5f5-4374-433f-a0e3-a71abc1dd30a" />

### 🏗️ 神經網路架構設計 (Model Architecture)

我們使用 PyTorch 的 `nn.Module` 建構多層感知機（MLP）網路：

```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

model = MNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
<img width="804" height="271" alt="image" src="https://github.com/user-attachments/assets/51443a55-267f-4f83-98f9-82adf7732b9f" />

### 🔄 模型訓練迴圈 (Training Loop)

使用 PyTorch 標準的 Forward & Backward 流程進行模型訓練，並紀錄歷程：

```python
EPOCHS = 20
history = {"loss": [], "accuracy": []}

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()   # 1. 梯度歸零
        outputs = model(images) # 2. 前向傳播
        loss = criterion(outputs, labels) # 3. 計算 Loss
        loss.backward()        # 4. 反向傳播
        optimizer.step()       # 5. 權重更新

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total
    history["loss"].append(epoch_loss)
    history["accuracy"].append(epoch_accuracy)
```
<img width="558" height="414" alt="image" src="https://github.com/user-attachments/assets/771c0a08-07e9-40c7-acfa-871cf4965468" />

### 🧪 模型評估與測試 (Model Evaluation)

在未參與訓練的測試集（Test Set）上測試模型的泛化能力：

```python
model.eval()

correct, total = 0, 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100.0 * correct / total
average_test_loss = test_loss / total
```
<img width="539" height="109" alt="image" src="https://github.com/user-attachments/assets/bb925332-d08c-49fd-bfef-21e7b8e0a99a" />

### 💾 模型儲存與檢查點保存 (Model Checkpointing)

為了防止 Colab 連線中斷導致訓練成果遺失，程式會自動將模型權重與完整歷史紀錄存入 Google Drive：

```python
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_model.pth")
torch.save(model.state_dict(), MODEL_PATH)

RESULT_PATH = os.path.join(RESULT_DIR, "training_results.pt")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": EPOCHS,
        "test_accuracy": test_accuracy,
        "test_loss": average_test_loss,
        "history": history
    },
    RESULT_PATH
)
```
<img width="834" height="101" alt="image" src="https://github.com/user-attachments/assets/0bc5b1c7-524b-49c2-9eff-d8a489d01aa9" />

### 📊 訓練歷程可視化 (Training Visualization)

使用 `matplotlib` 將 20 個 Epoch 的訓練 Loss 與 Accuracy 畫成雙子圖，並自動匯出至 Google Drive：

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 繪製 Training Loss 曲線
axes[0].plot(range(1, EPOCHS + 1), history["loss"], marker="o")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

# 繪製 Training Accuracy 曲線
axes[1].plot(range(1, EPOCHS + 1), history["accuracy"], marker="o", color="green")
axes[1].set_title("Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(True)

plt.tight_layout()
FIGURE_PATH = os.path.join(RESULT_DIR, "training_curves.png")
plt.savefig(FIGURE_PATH, dpi=150)
plt.show()
```
<img width="1153" height="424" alt="image" src="https://github.com/user-attachments/assets/60966049-dc83-44d5-a2e6-b2a78a905197" />

### 🖼️ 直觀模型推論與展示 (Model Inference & Visualization)

抽樣 10 張測試集圖片進行實時預測，並用顏色標示預測結果（綠色代表正確、紅色代表錯誤）：

```python
model.eval()

images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images[:10])
    predictions = outputs.argmax(dim=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].cpu().squeeze(), cmap="gray")
    actual = labels[i].cpu().item()
    predicted = predictions[i].cpu().item()
    
    ax.set_title(f"True: {actual} | Pred: {predicted}", color="green" if actual == predicted else "red")
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "prediction_examples.png"), dpi=150)
plt.show()
```
<img width="1127" height="664" alt="image" src="https://github.com/user-attachments/assets/8281f374-049e-412d-985f-e77fef61dd50" />
