##这里说明为了适应3d模式做的改变，以便debug
- 目录data:参见(data/change_data.txt)[data/change_data.txt]
- 无变化:`custom_latex.py`, `logger.py`, `main.py`, `main_dev.py`, `type_hints.py`
- +- `problem.py` #坐标传递优化,对应data/point.py变化(详情参见data/change_data.txt)
- + `api.py` #添加3d问题支持

```
from problem_3d import Problem_3d 
```
- + `problem_3d.py` ← `problem.py` #3d问题支持
添加函数:

```
def _get_vec_angle(self, v1: Matrix, v2: Matrix) -> Expr:
        """"向量夹角余弦
        二面角和(平面)角可以直接应用
        直线所成角和直线与平面所成角应用绝对值
        """
def _get_n_vec(self, name: str) -> Expr:
        """a×b可作为平面法向量
        大小为(向量围成的)平行四边形面积
        """
def _get_plp_angle(self, plp: str) -> Expr:
        """二面角:A(p)-BC(l)-D(p)
        传入plp时'-'应省略,即ABCD
        """
def _get_ll_angle(self, l1: str, l2: str) -> Expr:
        "(异面)直线所成角"
def _get_lp_angle(self, l: str, p: str) -> Expr:
        "直线与平面所成角"
def _get_pp_angle(self, p1: str, p2: str) -> Expr:
        "平面与平面所成角"
```
- `vec_parse_utils.py`:
- +- `mark_vec_coord`函数:#支持3d模式

```
def mark_vec_coord(expr: str) -> str:
    """
    用 ``Matrix([])`` 标记表达式中向量的坐标表示
    现支持多维向量
    """
    pattern = r'(?<!Matrix)\(([\d,],[\d,]+)\)'
    result = re.sub(pattern,r'Matrix([\1])',expr)
    return result
```
- + class `Infix`:#**添加提示**:中缀运算符用法
- + (`cross`)叉乘中缀运算符支持

```
cross = Infix(lambda a, b: a.cross(b))
```