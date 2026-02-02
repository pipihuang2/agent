# 代码设计模式入门

## 为什么要写 `get_mysql_manager()` ？

### 直接看对比

```python
# ❌ 方式 1: 每次都创建新对象
manager1 = MySQLQueryManager()  # 创建连接
manager2 = MySQLQueryManager()  # 又创建一个连接
manager3 = MySQLQueryManager()  # 又创建一个连接
# 问题: 创建了 3 个数据库连接，浪费资源！

# ✅ 方式 2: 用 get_mysql_manager() 获取
manager1 = get_mysql_manager()  # 第一次调用，创建对象
manager2 = get_mysql_manager()  # 返回同一个对象
manager3 = get_mysql_manager()  # 还是同一个对象
# 好处: 始终只有 1 个数据库连接
```

---

## 这叫什么设计模式？

### 1. 单例模式 (Singleton Pattern)

**目的**: 确保一个类只有一个实例

**代码解析**:
```python
# 全局变量，存储唯一的实例
_mysql_manager: Optional[MySQLQueryManager] = None

def get_mysql_manager() -> MySQLQueryManager:
    global _mysql_manager

    # 如果还没创建，就创建一个
    if _mysql_manager is None:
        _mysql_manager = MySQLQueryManager()

    # 返回这个唯一的实例
    return _mysql_manager
```

**图解**:
```
第一次调用 get_mysql_manager():
┌─────────────────────────────┐
│  _mysql_manager = None      │
│         ↓                   │
│  创建 MySQLQueryManager()   │
│         ↓                   │
│  _mysql_manager = 新对象    │
│         ↓                   │
│  返回 _mysql_manager        │
└─────────────────────────────┘

第二次调用 get_mysql_manager():
┌─────────────────────────────┐
│  _mysql_manager 已存在      │
│         ↓                   │
│  直接返回 _mysql_manager    │
└─────────────────────────────┘
```

---

### 2. 为什么需要单例？

**场景 1: 数据库连接**
```python
# 数据库连接是"昂贵"的资源
# 每次连接都要: 建立网络连接 → 身份验证 → 分配资源
# 如果每次都新建连接，会很慢，而且可能耗尽连接数
```

**场景 2: 配置对象**
```python
# 配置只需要读取一次
# 所有地方都应该用同一份配置
settings = get_settings()  # 全局共享同一个配置
```

**场景 3: 日志记录器**
```python
# 日志器应该是同一个，才能统一管理
logger = get_logger()
```

---

## 完整的代码结构

```python
"""mysql_query.py 的结构"""

from typing import Optional

# ============================================
# 第一部分: 定义类 (Class)
# ============================================
class MySQLQueryManager:
    """这是一个类，定义了"数据库管理器"应该有什么功能"""

    def __init__(self):
        """初始化：创建数据库连接"""
        self._engine = None  # 数据库引擎

    def test_connection(self):
        """功能1: 测试连接"""
        pass

    def list_tables(self):
        """功能2: 列出所有表"""
        pass

    def execute_query(self, query: str):
        """功能3: 执行查询"""
        pass


# ============================================
# 第二部分: 全局变量 (存储单例)
# ============================================
_mysql_manager: Optional[MySQLQueryManager] = None


# ============================================
# 第三部分: 获取函数 (单例模式)
# ============================================
def get_mysql_manager() -> MySQLQueryManager:
    """获取全局唯一的 MySQL 管理器实例"""
    global _mysql_manager
    if _mysql_manager is None:
        _mysql_manager = MySQLQueryManager()
    return _mysql_manager
```

---

## 你需要学什么？

### 学习路径（按顺序）

```
1. Python 基础
   ├── 变量、函数、类 (class)
   ├── 模块和包 (import)
   └── 类型提示 (Optional, list, dict)

2. 面向对象编程 (OOP)
   ├── 类和对象的区别
   ├── __init__ 方法
   ├── self 是什么
   └── 实例方法 vs 类方法

3. 设计模式（入门级）
   ├── 单例模式 (Singleton) ← 你问的这个
   ├── 工厂模式 (Factory)
   └── 依赖注入 (Dependency Injection)

4. Python 进阶
   ├── 装饰器 (@decorator)
   ├── 异步编程 (async/await)
   └── 上下文管理器 (with)
```

### 推荐资源

| 主题 | 推荐资源 |
|------|---------|
| Python 基础 | 《Python Crash Course》或 菜鸟教程 |
| 面向对象 | 《流畅的Python》第 8-11 章 |
| 设计模式 | 《Head First 设计模式》(有 Python 版) |
| 实际项目 | 看开源项目代码，比如 FastAPI |

---

## 动手练习

### 练习 1: 理解单例

```python
# 运行这段代码，观察输出

class Dog:
    def __init__(self, name):
        self.name = name
        print(f"创建了一只狗: {name}")

# 不用单例
dog1 = Dog("旺财")  # 输出: 创建了一只狗: 旺财
dog2 = Dog("小黑")  # 输出: 创建了一只狗: 小黑
print(dog1 is dog2)  # False，两个不同的对象

# 用单例
_global_dog = None

def get_dog():
    global _global_dog
    if _global_dog is None:
        _global_dog = Dog("单例狗")
    return _global_dog

dog3 = get_dog()  # 输出: 创建了一只狗: 单例狗
dog4 = get_dog()  # 没有输出！因为没有创建新对象
print(dog3 is dog4)  # True，同一个对象
```

### 练习 2: 自己写一个单例

```python
# 试着为 "配置管理器" 写一个单例

class ConfigManager:
    def __init__(self):
        print("读取配置文件...")
        self.debug = True
        self.database_url = "localhost:3306"

# TODO: 写 _config_manager 和 get_config_manager()
# 你的代码:
_config_manager = None

def get_config_manager():
    # 补全这里
    pass
```

---

## 总结

| 问题 | 答案 |
|------|------|
| 为什么要 `get_mysql_manager()` | 实现单例模式，确保全局只有一个数据库连接 |
| 这是什么设计模式 | 单例模式 (Singleton Pattern) |
| 好处是什么 | 节省资源、共享状态、统一管理 |
| 我要学什么 | Python 基础 → 面向对象 → 设计模式 |
