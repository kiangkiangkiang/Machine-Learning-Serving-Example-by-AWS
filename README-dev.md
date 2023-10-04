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



# 作法
- 把所有變數都 hash （不管連續或離散）
- if 一個欄位可存在多標籤時：找到最多標籤的那個，例如第 37 個 obs 擁有最多標籤 7 個，其他可能只有 2, 3 個，就 default 增加到 7 個欄位，有類別的就用編碼存，沒類別就存 0。

1. 把所有變數全都編碼，可以 one-hot 或是 hash
2. 定義 user 屬性類別， items 屬性類別，例如 性別, 年齡屬於同一個屬性，價錢, 評分是同一個屬性，user, item 分成兩個 array 存
3. 每個 list 內的子 list 代表各自屬性，例如 user[0]代表性別年齡（batch_size, 2），user[1]代表 ID居住地身高體重（batch_size, 4）
4. 定義 開頭 embedding size，有 k 個屬性類別，embedding size = k * default_embedding_size
5. 先過 Embedding，同樣屬性的可以用 sum or average 壓成同樣維度，例如過 user[0] Embedding大小變成（batch_size, 2, default_embedding_size），則 axis 1 就 sum 起來，保持最後為度都是（batch_size, default_embedding_size）
6. 最後 user, item 個別 concat
7. 過 MLP，第 4 點已經定義了開頭 embedding size，因此第一層 MLP一定是（開頭 embedding size, 後續的 embedding size）
8. user, item 都過後就 cos. sim
9. 




