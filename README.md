# LeRobot 数据集可视化与分析工具

面向一份 LeRobot 3.0 人体第一视角手部追踪数据集的 Episode 可视化与数据分析工具。

## 环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

依赖直接写在 `pyproject.toml` 里（`pyarrow`、`av`、`flask`）。

数据集本身不在仓库里（机密数据，已在 `.gitignore` 中排除），需要用凭证从
R2 对象存储同步一次到本地代码文件夹./data/raw下：

```bash
aws s3 sync s3://<bucket> ./data/raw \
  --endpoint-url <r2-endpoint> \
  --region auto
```

应用运行时只读本地 `data/raw/`，不做任何运行时的远程/网络访问。

## 启动方式

```bash
python scripts/analyze_dataset.py --dataset-root data/raw   # 写入 artifacts/analysis/summary.json
python scripts/run_visualizer.py --dataset-root data/raw
```

然后打开：
- `http://127.0.0.1:5000/` — **Viewer**

  支持浏览 Episode，并按 frame / timestamp 导航。页面围绕统一时间轴组织：
  
  - **Episode Navigation：** 左侧选择 Episode，并查看任务信息。
  - **Video & 3D Scene：** 顶部同步展示第一视角视频、3D 手部轨迹和相机位姿。
  - **Charts：** 展示手部运动、姿态变化等时序指标。
  - **Inspector：** 展示当前帧的 metadata、action、pose、track 等字段及原始数据。

  视频、3D 场景、图表和逐帧数据共享同一 frame / timestamp 时间轴并保持同步。
- `http://127.0.0.1:5000/analysis.html` — **Analysis**

  分析页面按 **Overview / Statistics / Relationships / Data Quality / Findings & Recommendations** 五个部分组织：

  - **Overview：** 汇总数据集规模、时长、FPS、robot type，以及关键字段的语义、shape 和坐标系；同时列出分析中的 assumptions / unknowns。
  - **Statistics：** 展示手部运动、姿态变化率、相机运动等指标的统计分布和分位数，并标记值得进一步检查的 high-motion regions。
  - **Relationships：** 分析不同数据模态之间的关系，包括 action 与 hand track 的时序偏移、video 与 dataset timestamp 同步，以及 camera motion 与 hand motion 的相关性。
  - **Data Quality：** 检查 schema 一致性、timestamp gap / duplicate、视频同步、quaternion 有效性、MANO betas 信息量，以及已验证的数据质量问题。
  - **Findings & Recommendations：** 将分析结果进一步整理为对训练、评估、特征选择和数据治理有价值的结论与建议。

  页面使用 **Verified Fact / Metadata Declaration / Inference / Hypothesis / Unknown** 标签区分直接观察到的事实、元数据声明、合理推断、待验证假设和未知项，并为关键发现提供 supporting evidence。

## 设计取舍

* **保持轻量依赖。** Flask + 原生 JS，无前端构建步骤；数据读取直接使用 PyArrow 解析 Parquet。项目验证过官方 `LeRobotDataset`，但当前数据集的 metadata schema 与官方 loader 的预期结构不兼容，因此采用轻量只读访问层。项目无需引入 `torch`，运行依赖保持在 `pyarrow`、`av` 和 `flask`。

* **访问层只读，且和上层逻辑分层解耦。** `dataset.py`（读取 parquet/视频
  路径）→ `semantics.py`（纯语义解释，不做任何 I/O）→ `metrics.py`（派生
  数值信号）→ `webapp.py`（唯一接触 Flask 的模块）。全链路没有任何写入
  原始数据的代码路径。

* **证据优先：区分“元数据声明”和“物理验证事实”。** 例如字段 dtype 的 metadata 声明和 Parquet 中实际读取到的物理类型会分别保留，不会互相覆盖。Analysis 页面也使用事实 / 元数据声明 / 推断 / 假设 / 未知标签区分关键结论的证据等级，避免把解释当作已验证事实。

* **分析结果预计算，而不是每次请求现算。** `analyze_dataset.py` 一次性跑完（包含解码视频做同步校验，比较慢）写入 JSON 产物，网页只读这份
  产物，不会在每次打开页面时重新计算。

## 已知限制

* **只有一个 episode。** 跨 episode 的检查（schema drift、按 episode 排序
  异常、任务均衡度）代码是通用实现，但在当前数据上只有 N=1，实际上是空跑，
  没有在多 episode 数据上验证过。

* **当前数据访问层针对本数据集 schema 做了适配。**
  由于当前 metadata schema 无法被测试过的官方 `LeRobotDataset` 直接读取，项目采用轻量只读访问层。该实现能够稳定支持当前数据，但并不是面向所有 LeRobot schema 的通用 loader；对于字段结构不同的数据集，需要增加适配代码。

* **不做完整的 MANO 3D 网格重建。** 需要有版权的 MANO body-model 资源，
  项目没有随附，因此直接把解析出来的 MANO 参数当数值信号使用，而不
  重建出手部网格。

* **3D 场景展示的是数据中记录的信号，不是标定过的世界坐标重建。**
  Metadata 里声明了相机内外参和畸变系数，但代码里没有任何投影或去畸变
  计算用到它们；3D 视图里坐标轴的翻转只是给 Three.js 用的渲染约定，不
  代表对物理坐标系做过标定级别的重建。

* **一些数据集本身没声明的约定，明确标注为"外部推测"，不当作事实。**
  比如 21 个手部关键点分别对应哪个部位这类信息数据集没有给出，只用于
  Viewer 里一个可选的叠加显示，并在页面上标注为未验证。

* **视频 / 时间戳同步是抽样校验，不是逐帧穷举。** 服务端会解码实际视频
  帧，和存储的时间戳比较，每个 episode 抽样约 15 帧，测得误差在微秒级
  （比一帧间隔小 4 个数量级），但这只覆盖抽样帧；浏览器端播放时用
  `timeupdate` 事件做最近帧查找，不是逐帧精确解码，不能当作严格的科学
  视频解码器。

* **本地开发服务器仅限单用户本地使用。** 因为数据集机密，Flask 默认绑定
  `127.0.0.1`，用的是 Werkzeug 内置开发服务器，不用于并发访问或生产
  环境。

* **数据来自私有对象存储，需要一次性同步。** 数据集存在 R2（S3 兼容）
  私有桶里，需要用凭证同步一次到本地；应用本身运行时不依赖网络或远程
  存储，只读本地已同步的文件。

## 可复现性

Visualizer 和 Analysis 页面基于同一份本地数据（`data/raw/`）。分析结果由 `scripts/analyze_dataset.py` 一次性计算并写入 `artifacts/analysis/summary.json`（该文件已加入 `.gitignore`），Analysis 页面只读取并展示该分析产物。

分析中的随机抽样使用固定随机种子，并有测试验证重复运行结果一致。页面展示的分析数值均来自生成的 JSON 产物，而非前端手写固定值。整条数据访问和分析链路不会修改 `data/raw/` 中的原始数据；数据或分析代码变化后，重新运行 `analyze_dataset.py` 即可刷新分析结果。
