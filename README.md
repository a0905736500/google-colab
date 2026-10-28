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

