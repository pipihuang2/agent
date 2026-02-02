# MCP 工具文档

本文档详细介绍了工业质量检测 Agent 中所有 MCP 工具的功能、输入参数和输出格式。

---

## 目录

- [数据查询工具](#数据查询工具)
- [数据分析工具](#数据分析工具)
- [图表生成工具](#图表生成工具)
- [MySQL 数据库工具](#mysql-数据库工具)

---

## 数据查询工具

### 1. `query_by_time_range`

**功能**: 按时间范围查询质量检测数据

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `start_time` | string | 否 | 开始时间，ISO 格式（如 `2024-01-01T00:00:00`） |
| `end_time` | string | 否 | 结束时间，ISO 格式 |
| `columns` | array[string] | 否 | 要返回的列名列表，不填返回全部列 |

**输出格式**:
```json
{
  "row_count": 50,
  "columns": ["timestamp", "product_id", "measurement_1", "temperature"],
  "data": [
    {"timestamp": "2024-01-01T00:00:00", "product_id": "P001", "measurement_1": 10.5, "temperature": 25.0},
    ...
  ]
}
```

---

### 2. `query_by_product`

**功能**: 按产品 ID 查询该产品的所有检测数据

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `product_id` | string | **是** | 产品 ID（如 `P001`） |
| `columns` | array[string] | 否 | 要返回的列名列表 |

**输出格式**:
```json
{
  "product_id": "P001",
  "row_count": 10,
  "columns": ["timestamp", "measurement_1", "pass_fail"],
  "data": [...]
}
```

---

### 3. `query_by_batch`

**功能**: 按批次 ID 查询该批次的所有检测数据

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `batch_id` | string | **是** | 批次 ID（如 `B001`） |
| `columns` | array[string] | 否 | 要返回的列名列表 |

**输出格式**:
```json
{
  "batch_id": "B001",
  "row_count": 10,
  "columns": ["timestamp", "product_id", "measurement_1"],
  "data": [...]
}
```

---

### 4. `get_data_summary`

**功能**: 获取数据源的概览信息

**输入参数**: 无

**输出格式**:
```json
{
  "total_rows": 100,
  "columns": ["timestamp", "product_id", "measurement_1", "measurement_2", "temperature", "humidity", "pass_fail", "batch_id"],
  "time_range": {
    "start": "2024-01-01 00:00:00",
    "end": "2024-01-05 03:00:00"
  },
  "unique_products": 10,
  "unique_batches": 10
}
```

---

### 5. `get_available_columns`

**功能**: 获取数据源中所有可用的列名

**输入参数**: 无

**输出格式**:
```json
{
  "columns": ["timestamp", "product_id", "measurement_1", "measurement_2", "temperature", "humidity", "pass_fail", "batch_id"]
}
```

---

## 数据分析工具

### 1. `statistical_analysis`

**功能**: 对数值数据进行描述性统计分析，可选包含过程能力分析（Cp/Cpk）

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | array[number] | **是** | 数值数组 |
| `include_capability` | boolean | 否 | 是否计算过程能力指标，默认 `false` |
| `usl` | number | 否 | 规格上限（计算 Cp/Cpk 时需要） |
| `lsl` | number | 否 | 规格下限（计算 Cp/Cpk 时需要） |

**输出格式**:
```json
{
  "count": 100,
  "mean": 10.25,
  "std": 0.15,
  "min": 9.8,
  "max": 10.7,
  "median": 10.24,
  "q1": 10.1,
  "q3": 10.4,
  "range": 0.9,
  "iqr": 0.3,
  "skewness": 0.12,
  "kurtosis": -0.05,
  "capability": {
    "cp": 1.33,
    "cpk": 1.25,
    "cpu": 1.30,
    "cpl": 1.25,
    "within_spec_percentage": 99.7
  }
}
```

**说明**:
- `capability` 字段仅当 `include_capability=true` 且提供了 `usl` 和 `lsl` 时返回
- Cp >= 1.33 表示过程能力良好
- Cpk >= 1.33 表示过程能力良好且居中

---

### 2. `anomaly_detection`

**功能**: 检测数据中的异常值，支持 Z-score 和 IQR 两种方法

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | array[number] | **是** | 数值数组 |
| `method` | string | 否 | 检测方法：`zscore` 或 `iqr`，默认 `zscore` |
| `threshold` | number | 否 | 阈值，zscore 默认 3.0，iqr 建议 1.5 |

**输出格式 (Z-score 方法)**:
```json
{
  "method": "z-score",
  "threshold": 3.0,
  "total_points": 100,
  "anomaly_count": 2,
  "anomaly_percentage": 2.0,
  "anomalies": [
    {"index": 45, "value": 15.2, "z_score": 3.5},
    {"index": 78, "value": 5.1, "z_score": -3.2}
  ]
}
```

**输出格式 (IQR 方法)**:
```json
{
  "method": "IQR",
  "threshold_multiplier": 1.5,
  "bounds": {
    "lower": 9.5,
    "upper": 11.0
  },
  "total_points": 100,
  "anomaly_count": 3,
  "anomaly_percentage": 3.0,
  "anomalies": [
    {"index": 23, "value": 8.9, "deviation": "below"},
    {"index": 67, "value": 11.5, "deviation": "above"}
  ]
}
```

---

### 3. `trend_analysis`

**功能**: 分析时间序列数据的趋势，包括线性回归和移动平均

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | array[number] | **是** | 按时间顺序排列的数值数组 |
| `timestamps` | array[string] | 否 | 对应的时间戳列表 |

**输出格式**:
```json
{
  "data_points": 100,
  "trend_direction": "increasing",
  "linear_trend": {
    "slope": 0.05,
    "intercept": 10.0,
    "r_squared": 0.85
  },
  "moving_average": {
    "window_size": 5,
    "values": [10.1, 10.15, 10.2, ...]
  },
  "period_comparison": {
    "first_half_mean": 10.1,
    "second_half_mean": 10.5
  }
}
```

**说明**:
- `trend_direction`: `increasing`（上升）、`decreasing`（下降）、`stable`（稳定）
- `r_squared`: 决定系数，越接近 1 表示线性趋势越明显

---

### 4. `control_chart_analysis`

**功能**: 进行控制图分析，计算控制限，检测失控点

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | array[number] | **是** | 数值数组 |
| `chart_type` | string | 否 | 控制图类型：`individuals`（I-MR图）或 `xbar`（X-bar图），默认 `individuals` |
| `subgroup_size` | integer | 否 | X-bar 图的子组大小，默认 5 |

**输出格式 (I-MR 图)**:
```json
{
  "chart_type": "Individuals (I-MR)",
  "center_line": 10.25,
  "ucl": 10.75,
  "lcl": 9.75,
  "mr_bar": 0.18,
  "sigma_estimate": 0.16,
  "out_of_control_points": [
    {"index": 45, "value": 10.9, "type": "out_of_limits"}
  ],
  "in_control": false
}
```

**输出格式 (X-bar 图)**:
```json
{
  "chart_type": "X-bar",
  "subgroup_size": 5,
  "num_subgroups": 20,
  "center_line": 10.25,
  "ucl": 10.55,
  "lcl": 9.95,
  "r_bar": 0.52,
  "subgroup_means": [10.2, 10.3, ...],
  "out_of_control_points": [],
  "in_control": true
}
```

---

## 图表生成工具

所有图表工具都会将图表保存为 PNG 文件，并返回文件路径。

### 通用输出格式

```json
{
  "success": true,
  "filepath": "./output/charts/line_chart_20240101_120000.png",
  "filename": "line_chart_20240101_120000.png",
  "base64": "iVBORw0KGgo..."
}
```

> 注：`base64` 字段仅当 `return_base64=true` 时返回

---

### 1. `generate_line_chart`

**功能**: 生成折线图，适用于时间序列或顺序数据

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | array[number] | **是** | 要绑制的数值列表 |
| `labels` | array[string] | 否 | X 轴标签 |
| `title` | string | 否 | 图表标题，默认 "Line Chart" |
| `xlabel` | string | 否 | X 轴标签，默认 "Index" |
| `ylabel` | string | 否 | Y 轴标签，默认 "Value" |
| `show_trend` | boolean | 否 | 是否显示趋势线，默认 `false` |

---

### 2. `generate_bar_chart`

**功能**: 生成柱状图，适用于类别数据对比

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | array[number] | **是** | 数值列表 |
| `labels` | array[string] | **是** | 类别标签 |
| `title` | string | 否 | 图表标题，默认 "Bar Chart" |
| `xlabel` | string | 否 | X 轴标签，默认 "Category" |
| `ylabel` | string | 否 | Y 轴标签，默认 "Value" |
| `horizontal` | boolean | 否 | 是否横向显示，默认 `false` |

---

### 3. `generate_scatter_plot`

**功能**: 生成散点图，适用于相关性分析

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `x_data` | array[number] | **是** | X 轴数值 |
| `y_data` | array[number] | **是** | Y 轴数值 |
| `title` | string | 否 | 图表标题，默认 "Scatter Plot" |
| `xlabel` | string | 否 | X 轴标签，默认 "X" |
| `ylabel` | string | 否 | Y 轴标签，默认 "Y" |
| `show_regression` | boolean | 否 | 是否显示回归线及 R²，默认 `false` |

---

### 4. `generate_heatmap`

**功能**: 生成热力图，适用于矩阵数据可视化

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | array[array[number]] | **是** | 二维数值数组 |
| `x_labels` | array[string] | 否 | 列标签 |
| `y_labels` | array[string] | 否 | 行标签 |
| `title` | string | 否 | 图表标题，默认 "Heatmap" |
| `colorbar_label` | string | 否 | 颜色条标签，默认 "Value" |

---

### 5. `generate_histogram`

**功能**: 生成直方图，适用于数据分布分析

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | array[number] | **是** | 数值列表 |
| `bins` | integer | 否 | 分箱数量，默认 20 |
| `title` | string | 否 | 图表标题，默认 "Histogram" |
| `xlabel` | string | 否 | X 轴标签，默认 "Value" |
| `ylabel` | string | 否 | Y 轴标签，默认 "Frequency" |
| `show_normal` | boolean | 否 | 是否叠加正态分布曲线，默认 `false` |
| `usl` | number | 否 | 规格上限，显示为垂直线 |
| `lsl` | number | 否 | 规格下限，显示为垂直线 |

---

### 6. `generate_control_chart`

**功能**: 生成控制图，带有 UCL、LCL 和中心线

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | array[number] | **是** | 数值列表 |
| `ucl` | number | **是** | 控制上限 (Upper Control Limit) |
| `lcl` | number | **是** | 控制下限 (Lower Control Limit) |
| `cl` | number | **是** | 中心线 (Center Line) |
| `title` | string | 否 | 图表标题，默认 "Control Chart" |
| `ylabel` | string | 否 | Y 轴标签，默认 "Value" |
| `labels` | array[string] | 否 | X 轴标签 |

**说明**: 超出控制限的点会以红色标注并显示数值

---

### 7. `generate_defect_heatmap`

**功能**: 在产品图像上生成缺陷热力图，显示缺陷密度分布

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `image_path` | string | **是** | 产品图像路径（支持 PNG、JPG 等） |
| `ng_coordinates` | array[object] | **是** | 缺陷坐标列表，每个对象包含 `x` 和 `y` |
| `title` | string | 否 | 图表标题，默认 "Defect Heatmap" |
| `sigma` | number | 否 | 高斯模糊参数，值越大热区越平滑，默认 20.0 |
| `alpha` | number | 否 | 热力图透明度 (0.0-1.0)，默认 0.6 |
| `colorbar_label` | string | 否 | 颜色条标签，默认 "Defect Density" |

**ng_coordinates 格式**:
```json
[
  {"x": 100, "y": 200},
  {"x": 150, "y": 250},
  {"x": 100, "y": 205}
]
```

**输出格式**:
```json
{
  "success": true,
  "filepath": "./output/charts/defect_heatmap_20240101_120000.png",
  "filename": "defect_heatmap_20240101_120000.png",
  "statistics": {
    "total_ng_points": 150,
    "valid_ng_points": 148,
    "image_size": {"width": 1920, "height": 1080},
    "max_density_location": {"x": 450, "y": 320}
  }
}
```

**说明**: 热力图颜色从透明 → 黄色 → 橙色 → 红色，红色区域表示缺陷密度最高

---

## MySQL 数据库工具

### 1. `mysql_test_connection`

**功能**: 测试 MySQL 数据库连接

**输入参数**: 无

**输出格式**:
```json
{
  "success": true,
  "message": "Database connection successful"
}
```

**错误输出**:
```json
{
  "success": false,
  "error": "Connection refused..."
}
```

---

### 2. `mysql_list_tables`

**功能**: 列出数据库中的所有表

**输入参数**: 无

**输出格式**:
```json
{
  "success": true,
  "tables": ["quality_data", "products", "batches", "defects"],
  "count": 4
}
```

---

### 3. `mysql_describe_table`

**功能**: 获取表的结构信息（列名、类型、键等）

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `table_name` | string | **是** | 表名 |

**输出格式**:
```json
{
  "success": true,
  "table": "quality_data",
  "columns": [
    {"name": "id", "type": "int(11)", "null": "NO", "key": "PRI", "default": null, "extra": "auto_increment"},
    {"name": "timestamp", "type": "datetime", "null": "NO", "key": "", "default": null, "extra": ""},
    {"name": "measurement", "type": "decimal(10,4)", "null": "YES", "key": "", "default": null, "extra": ""}
  ],
  "column_count": 3
}
```

---

### 4. `mysql_execute_query`

**功能**: 执行 SQL 查询（仅限 SELECT 语句，出于安全考虑）

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `query` | string | **是** | SQL SELECT 语句 |
| `limit` | integer | 否 | 最大返回行数，默认 1000 |

**示例输入**:
```json
{
  "query": "SELECT * FROM quality_data WHERE status = 'FAIL' ORDER BY timestamp DESC",
  "limit": 100
}
```

**输出格式**:
```json
{
  "success": true,
  "data": [
    {"id": 1, "timestamp": "2024-01-01 10:00:00", "measurement": 10.5, "status": "FAIL"},
    ...
  ],
  "row_count": 50,
  "columns": ["id", "timestamp", "measurement", "status"]
}
```

**安全限制**: 只允许 SELECT 查询，其他操作（INSERT、UPDATE、DELETE）会被拒绝

---

### 5. `mysql_query_table`

**功能**: 简化的表查询，支持过滤和排序

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `table_name` | string | **是** | 表名 |
| `columns` | array[string] | 否 | 要查询的列，不填返回全部 |
| `where` | string | 否 | WHERE 条件（不含 WHERE 关键字） |
| `order_by` | string | 否 | ORDER BY 子句（不含 ORDER BY 关键字） |
| `limit` | integer | 否 | 最大返回行数，默认 1000 |

**示例输入**:
```json
{
  "table_name": "quality_data",
  "columns": ["timestamp", "measurement", "status"],
  "where": "status = 'FAIL' AND temperature > 30",
  "order_by": "timestamp DESC",
  "limit": 100
}
```

**输出格式**:
```json
{
  "success": true,
  "table": "quality_data",
  "data": [...],
  "row_count": 25,
  "columns": ["timestamp", "measurement", "status"]
}
```

---

### 6. `mysql_get_table_stats`

**功能**: 获取表中数值列的统计摘要

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `table_name` | string | **是** | 表名 |
| `column` | string | **是** | 数值列名 |

**输出格式**:
```json
{
  "success": true,
  "table": "quality_data",
  "column": "measurement",
  "statistics": {
    "count": 1000.0,
    "mean": 10.25,
    "min": 9.5,
    "max": 11.2,
    "std": 0.35
  }
}
```

---

### 7. `mysql_get_column_values`

**功能**: 获取某列的所有值（用于传递给分析工具）

**输入参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `table_name` | string | **是** | 表名 |
| `column` | string | **是** | 列名 |
| `limit` | integer | 否 | 最大返回数量，默认 10000 |

**输出格式**:
```json
{
  "success": true,
  "table": "quality_data",
  "column": "measurement",
  "values": [10.2, 10.3, 10.1, 10.5, ...],
  "count": 1000
}
```

**使用场景**: 获取数据后传递给 `statistical_analysis` 或 `anomaly_detection` 等分析工具

---

## 工具调用示例

### 示例 1: 查询数据并进行统计分析

```python
# 1. 先获取数据
result = query_by_time_range(
    start_time="2024-01-01T00:00:00",
    end_time="2024-01-31T23:59:59",
    columns=["measurement_1"]
)

# 2. 提取数值进行分析
data = [row["measurement_1"] for row in result["data"]]
stats = statistical_analysis(
    data=data,
    include_capability=True,
    usl=10.5,
    lsl=9.5
)
```

### 示例 2: 从数据库分析数据并生成控制图

```python
# 1. 从 MySQL 获取数据
values = mysql_get_column_values(
    table_name="quality_data",
    column="measurement"
)

# 2. 进行控制图分析
analysis = control_chart_analysis(
    data=values["values"],
    chart_type="individuals"
)

# 3. 生成控制图
chart = generate_control_chart(
    data=values["values"],
    ucl=analysis["ucl"],
    lcl=analysis["lcl"],
    cl=analysis["center_line"],
    title="质量测量控制图"
)
```

### 示例 3: 生成缺陷热力图

```python
# 假设有一组缺陷坐标
defects = [
    {"x": 100, "y": 200},
    {"x": 105, "y": 198},
    {"x": 450, "y": 300},
    # ... 更多坐标
]

result = generate_defect_heatmap(
    image_path="./product_image.png",
    ng_coordinates=defects,
    title="产品缺陷分布热力图",
    sigma=25.0,
    alpha=0.7
)
```

---

## 工具分类总结

| 类别 | 工具数量 | 主要用途 |
|------|---------|----------|
| 数据查询 | 5 | 从内置数据源查询检测数据 |
| 数据分析 | 4 | 统计分析、异常检测、趋势分析、控制图分析 |
| 图表生成 | 7 | 折线图、柱状图、散点图、热力图、直方图、控制图、缺陷热力图 |
| MySQL 数据库 | 7 | 数据库连接、表操作、数据查询 |
| **总计** | **23** | |
