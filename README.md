# LeRobot 数据集可视化与分析工具

面向一个 LeRobot 3.0 数据集（人体第一视角手部追踪）的
episode 可视化工具，以及一个数据集级别的分析页面。

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
- `http://127.0.0.1:5000/` — Viewer（视频、3D 场景、逐帧 metadata / action /
  pose / track 字段、时序图表、frame/timestamp 导航）；
- `http://127.0.0.1:5000/analysis.html` — Analysis 页面（数据结构、统计分布、
  跨模态关系、数据质量与结论）。

## 设计取舍

* **轻量级技术栈，不引入额外依赖。** Flask + 原生 JS、无构建步骤；数据
  读取直接用 PyArrow 解析 parquet。项目评估过直接使用官方 `LeRobotDataset`
  loader，但所提供数据集的 metadata 与测试过的官方 loader 路径不完全
  兼容，因此改为自研的轻量只读访问层——整个项目也因此不需要 `torch`，
  依赖始终保持在 `pyarrow` + `av` + `flask` 三个包，安装和复现成本很低。

* **访问层只读，且和上层逻辑分层解耦。** `dataset.py`（读取 parquet/视频
  路径）→ `semantics.py`（纯语义解释，不做任何 I/O）→ `metrics.py`（派生
  数值信号）→ `webapp.py`（唯一接触 Flask 的模块）。全链路没有任何写入
  原始数据的代码路径。

* **证据优先：区分"元数据声明"和"物理验证事实"。** 例如某个字段的
  dtype，metadata 里声明的类型和从 parquet 实际读到的物理类型会分别保留、
  分别暴露，不会互相覆盖。Analysis 页面上的每一条非平凡结论也都标注了
  事实 / 元数据声明 / 推断 / 假设 / 未知 五档标签，避免把解释当结论讲。

* **分析结果预计算，而不是每次请求现算。** `analyze_dataset.py` 一次性
  跑完（包含解码视频做同步校验，比较慢）写入 JSON 产物，网页只读这份
  产物，不会在每次打开页面时重新计算。

## 已知限制

* **只有一个 episode。** 跨 episode 的检查（schema drift、按 episode 排序
  异常、任务均衡度）代码是通用实现，但在当前数据上只有 N=1，实际上是空跑，
  没有在多 episode 数据上验证过。

* **是针对这份数据集 schema 定制的分析器，不是通用 LeRobot 分析器。**
  手部 track、MANO 参数、四元数关键点这些字段的解释都是按当前数据集的
  具体格式写的，换一个字段结构明显不同的数据集需要改代码。

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

Visualizer 和 Analysis 页面读取的是同一份本地数据（`data/raw`）。分析
结果由 `scripts/analyze_dataset.py` 一次性计算并写入 `artifacts/analysis/
summary.json`（该文件已加入 `.gitignore`），网页只是展示这份产物；随机
抽样用固定随机种子，有专门测试验证两次运行结果完全一致。Analysis 页面上
引用的每一个数字都是从这份 JSON 实时计算展示的，没有手写死的数字。整条
链路只读，不会修改 `data/raw` 里的原始数据；改动数据或分析代码后，重新
跑一遍 `analyze_dataset.py` 即可刷新结果。
