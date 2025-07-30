##这里说明为了适应3d模式做的改变，以便debug

- 目录data:参见[data/change\_data.txt](data/change_data.txt)
- 无变化:`api.py`, `custom_latex.py`, `logger.py`, `main.py`, `main_dev.py`, `type_hints.py`
- +- `problem.py` #坐标传递优化,对应data/point.py变化(详情参见data/change\_data.txt)
- \+ `api_3d.py` ← `api.py` #3d问题支持

```python
from logger import frontend_logger
from problem_3d import Problem_3d
class API_3d:
    problem_3d = Problem_3d()
    logger_3d = frontend_logger
api_3d = API_3d()
```
- \+ `main_3d.py` ← `main.py`
- \+ `main_dev_3d.py` ← `main_dev.py`
- \+ `problem_3d.py` ← `problem.py` #3d问题支持
1. 改变函数:

```python
def _get_distance(self, name: str) -> Expr:
    + "向量大小=端点距离"
    +- return self._get_vec(name).norm()# 不用距离公式，直接获取向量大小
def _get_angle(self, name: str) -> Expr:
    + "a·b=|a||b|cos θ:∠ABC"
    v1 = self._get_vec(name[1::-1])
    v2 = self._get_vec(name[1:])
    + cos0 = self._get_vec_angle(v1,v2)
    +- return acos(cos0) #折中方法(中间函数法)，区分直线所成角表示
def _get_triangle_area(self, name: str) -> Expr:
    "叉乘大小的一半可表示三角形面积"
    +- # 叉乘表示(真香.jpg 2233...
    Sp = self._get_n_vec(name).norm()
    return Sp/Integer(2)
```
2. 添加函数:

```python
def _get_vec_angle(self, v1: Matrix, v2: Matrix) -> Expr:
    """"
    中间函数:向量夹角余弦
    !!!只能在内部使用!!!
    二面角和(平面)角可以直接应用
    直线所成角和直线与平面所成角应用绝对值
    """
    return v1.dot(v2) / (v1.norm() * v2.norm())
def _get_n_vec(self, name: str) -> Expr:
    """
    a×b可作为平面法向量(sp)ABC
    大小为(向量围成的)平行四边形面积
    """
def _get_plp_angle(self, plp: str) -> Expr:
    """
    二面角:A(p)-BC(l)-D(p)
    传入plp时'-'应省略,即(pang)ABCD
    cosθ=±cos<n1,n2>，异侧同角，同侧补角
    下面n1,n2为平面的异侧方向->θ=<n1,n2>
    """
def _get_ll_angle(self, l1: str, l2: str) -> Expr:
    """"
    (异面)直线(AB)与(CD)所成角(ll)
    传入ll时，应为(ll)AB/CD
    cosθ=|cos<v1,v2>|
    """
def _get_lp_angle(self, l: str, p: str) -> Expr:
    """
    直线AB与平面(sp)ABC所成角(lp)
    (lp)AB/spABC
    sinθ=|cos<v,n>|
    """
def _get_pp_angle(self, p1: str, p2: str) -> Expr:
    """
    平面(sp)ABC与平面(sp)BCD所成角(pp)
    (pp)spABC/spBCD
    cosθ=|cos<n1,n2>|
    """
def _get_distance_pl(self, p: str, l: str) -> Expr:
    """
    点A到直线BC(底a)的距离(d)
    Sp=d*a=|va×vb|
    传入:dAtBC
    """
def _get_distance_pp(self, point: str, plane: str) -> Expr:
    """
    点A到平面BCD的距离(h1)
    点A到BC距离(h2)
    平面ABC与平面BCD所成角(θ)
    h1=h2*sinθ
    传入:dAtBCD
    """
def _get_volume(self, trip: str) -> Expr:
    """
    三棱锥(四面体)A-BCD体积
    传入trip时应省略'-',即(v)ABCD
    V=1/3*d*S
    """
```
- `vec_parse_utils.py`:
1. +- `mark_vec_coord`函数:#支持3d模式

```python
def mark_vec_coord(expr: str) -> str:
    """
    用 ``Matrix([])`` 标记表达式中向量的坐标表示
    现支持多维向量
    """
    pattern = r'(?<!Matrix)\(([\d,],[\d,]+)\)'
    result = re.sub(pattern,r'Matrix([\1])',expr)
    return result
```
2. \+ class `Infix`:#**添加提示**:中缀运算符用法
3. \+ (`cross`)叉乘中缀运算符支持

```python
cross = Infix(lambda a, b: a.cross(b))
```