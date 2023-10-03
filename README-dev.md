#### 目標

建立一套推薦系統（暫時不處理冷啟動，冷啟動可能會設計一些 UI 讓使用者點選幾個興趣，之後推薦）

想像場景：當使用者點開首頁時會自動觸發 trigger，收集使用者當前資訊後進行推薦

輸入：使用者 ID，


1. 找合適推薦系統的 Dataset (Tabular)
2. 訓練一個推薦系統模型 (連動 S3，自己建一些資料庫)
3. 想一下要怎麼做到CI/CD
4. 想辦法包成 container 部署上去
5. 想一下要有幾個端點 幾個計算機器
6. 想一下要怎麼做 load balancer
7. 設計一些 function 
   1. 一次訓練當日全部資料（Sample 機制）
   2. 一次訓練部分資料然後部署
   3. 推論

# 實作流程

Serving 時，
召回原則：只有當前 User 會跑 Inference，其他都是 Dict 查找。
給定 User ID，自動去資料庫撈相關資料。
*粗排 (當召回的數量龐大，需要一個模型輔助收斂，召回數量小，此環節可省略)*

### 1. 離線訓練（使用歷史數據）

1. 訓練雙塔模型（召回）
2. 建立 UserCF（召回）
3. 建立 ItemCF（召回）
4. Item 分群（召回）
5. 訓練 MMoE 模型（精排）

### 2. 建 Database

1. Raw User Feature Data（線上需要，用來過一次 User Tower，給定 User id，撈出所需要的 Feature）
2. Raw Item Feature Data（用來訓練 Item Embedding DB，線上不需要）
3. CF DB（共 3 張 Table）：
   1. ItemCF (實作上給定用戶 id 找到前 n 個用戶喜歡的物品，再根據 n 個物品找到前 k 個相似物品，共召回 nk 個物品，取前 m 個作為最終召回結果)  
      1. Table. 1：用戶興趣表：(用戶 id, item list)
         - eg. : user_dict[”1820718”] == [(Item5, 3), (Item3, 1), (Item6, 0)]
      2. Table. 2: 物品相似度表：（物品 id, item list）
         - eg. : item_dict["80227"] == [(Item1, 0.3), (Item8, 0.8), (Item2, 0.1)]
   2. UserCF (實作上給定用戶 id 找到前 n 個相似用戶，再根據 Table. 1 找到 k 個相似用戶喜歡的物品，共召回 nk 個物品，取前 m 個作為最終召回結果) 
      1. Table. 3：用戶相似列表：（用戶 id, 用戶 list）
         - eg. : user_dict[”1820718”] == [(User5, 0.3), (User3, 0.1), (User6, 0.9)]
4. Item Embedding DB:
   1. Table.1 : Raw Item Embedding (item id, Embedding)
   2. Table.2 : Item cluster Embedding (cluster kernal Item id, cluster member item id)
      - 線下先分群，線上時先跟質心算相似度，找到最近的質心，再跟質心內的所有成員算相似度（用於雙塔召回）



# 步驟

## 建立資料集

``` Shell
sh get_data.sh
```

```Shell
pip install git+https://github.com/PaddlePaddle/PaddleRec.git
```

```python
python -m rec-sys

```
