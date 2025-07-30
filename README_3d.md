> 作为高中三大禁术之一是必扬的(bushi[/doge]
> <p align="right">——编者按</p>

# 几何计算器 2

## 其它语言

* [English (US)](README.en.md)
* [简体中文](README.md)

借助计算机的强大算力，使用解析几何暴力计算几何问题！

- [使用文档](frontend/src/pages/docs.md)
- [关于 几何计算器 2](frontend/src/pages/about.md)
### 改变:这里说明为了适应3d模式做的改变

1. 后端`bachend/src/`目录:参见目录下[change_src.md文件](backend/src/change\_src.md)
2. **新三角形写法**:`trABC` #这是为了区分平面写法
3. **新增平面写法**:`spABC` #实际表示平面法向量
### 关于三维(3d)模式支持

1. 基本用法不变,平面写法仅支持三点(如`spABC`)
2. 三维叉乘(**cross**):结果为一个三维向量(可以当作平面法向量)
如_AB_×_AC_写成`vecAB cross vecAC`，不过要注意:
- 叉乘只能用于三维空间的向量
- 两个平行向量叉乘结果为零向量,不能作为法向量
#### 启动后端(选择其中一个)

1. 在 `backend/` 目录下运行 `main_dev.py`，这样整个2d项目就启动完成了。
2. 在 `backend/` 目录下运行 `main_dev_3d.py`，这样整个3d项目就启动完成了。