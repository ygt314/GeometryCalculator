from custom_latex import override_latex
override_latex()
from typing import Never, Optional, Callable
import re
import functools
from abc import ABC, abstractmethod
from collections import deque
import pickle
from sympy import Symbol, Expr, simplify, Eq, Line3D, solve, Point3D, Matrix, latex, Abs, sqrtdenest
from sympy import sqrt, sin, asin, cos, acos, tan, pi, Integer  # noqa
from sympy.logic.boolalg import BooleanTrue, BooleanFalse
from webview import windows, SAVE_DIALOG, OPEN_DIALOG
from data import MathObj, GCSymbol_3d, GCPoint_3d, Cond_3d, to_raw_latex_3d
from type_hints import DomainSettings, LatexItem
from vec_parse_utils import mark_vec_coord, dot, cross

x = Symbol('x', real=True)
y = Symbol('y', real=True)
z = Symbol('z', real=True)
# 希腊字母的英文拼写（除 pi 外）
VALID_GREEK_SPELLINGS = [
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron',
    'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
]

def track_requirement_3d(func):
    """
    在执行访问数学对象的函数时，
    追踪记录它访问了谁
    """
    @functools.wraps(func)
    def wrapper(self: 'Problem_3d', name: str):
        self.requirements_tracker.add(self.math_objs[name])
        return func(self, name)

    return wrapper

class AddCond_3d(ABC):
    def __init__(self, op: str):
        """
        装饰添加条件的方法，在该装饰器内实现把用户输入的表达式解析并拼接成 LaTeX 作为该条件的 ``id``，并添加条件
        这样被装饰方法只要专注于给出解析的方程（组）就行了（这里还会对每个方程进行化简并过滤掉 True）
        :param op: 该种类条件的符号（可能是放在中间的关系符，也可能是放在前面的图形类型）
        """
        self.op = op

    @abstractmethod
    def get_raw_latex(self, *args) -> str:
        """给出原始形式的 LaTeX"""
        ...

    def __call__(self, func: Callable[['Problem_3d', str, str], list[Eq]]):
        def wrapper(problem: 'Problem_3d', *args) -> None | Never:
            raw_latex = self.get_raw_latex(*args)
            # 化简方程（组）并过滤 True
            eqs = []
            for eq in func(problem, *args):
                eq = simplify(eq)
                if isinstance(eq, BooleanFalse):
                    raise ValueError('该条件不可能成立！')
                if not isinstance(eq, BooleanTrue):
                    eqs.append(eq)
            if len(eqs) == 0:
                raise ValueError('该条件一定成立，不需要添加')
            problem.add_cond(Cond_3d(raw_latex, eqs))

        return wrapper

class AddBinCond_3d(AddCond_3d):
    def get_raw_latex(self, input1: str, input2: str) -> str:
        return f'{to_raw_latex_3d(input1)} {self.op} {to_raw_latex_3d(input2)}'

class AddUnaryCond_3d(AddCond_3d):
    def get_raw_latex(self, input1: str) -> str:
        return f'{self.op} {input1}'

class Problem_3d:
    def __init__(self):
        self.math_objs: dict[str, MathObj] = {}
        self.symbol_names: list[str] = []
        self.point_names: list[str] = []
        self.cond_ids: list[str] = []
        # 用于临时存放正在添加的新对象依赖哪些对象
        self.requirements_tracker: set[MathObj] = set()

    def _add_math_obj(self, obj: MathObj) -> None:
        "添加数学对象，并添加它的依赖关系"
        self.math_objs[obj.id] = obj
        # 添加依赖关系并清空追踪器
        for requirement in self.requirements_tracker:
            requirement.add_required_by(obj)
        self.requirements_tracker.clear()

    def add_cond(self, cond: Cond_3d) -> None:
        """
        添加条件并把 ``id`` 加到列表里
        注意：此处函数名不以下划线开头，是为了方便 Python 中的外部装饰器调用这个方法，该方法不在 TS 中声明暴露
        """
        self._add_math_obj(cond)
        self.cond_ids.append(cond.id)

    @track_requirement
    def _get_sp_symbol(self, name: str) -> Symbol:
        return self.math_objs[name].sp_symbol  # type: ignore

    @track_requirement
    def _get_x_of(self, name: str) -> Expr:
        return self.math_objs[name].x  # type: ignore

    @track_requirement
    def _get_y_of(self, name: str) -> Expr:
        return self.math_objs[name].y  # type: ignore

    @track_requirement
    def _get_z_of(self, name: str) -> Expr:
        return self.math_objs[name].z  # type: ignore

    @track_requirement
    def _get_sp_point(self, name: str) -> Point3D:
        return self.math_objs[name].sp_point  # type: ignore

    def _get_line(self, name: str) -> Line3D:
        "两点定直线,似乎没什么用(用向量足够了(/doge)"
        p1 = self._get_sp_point(name[0])
        p2 = self._get_sp_point(name[1])
        return Line(p1, p2)

    def _get_vec(self, name: str) -> Matrix:
        "获取向量（实际上是个矩阵）"
        initial = self._get_sp_point(name[0])
        terminal = self._get_sp_point(name[1])
        return Matrix(terminal-initial)

    def _get_distance(self, name: str) -> Expr:
        "向量大小=端点距离"
        return self._get_vec(name).norm()

    def _get_vec_angle(self, v1: Matrix, v2: Matrix) -> Expr:
        """"
        中间函数:向量夹角余弦
        !!!只能在内部使用!!!
        二面角和(平面)角可以直接应用
        直线所成角和直线与平面所成角应用绝对值
        """
        return v1.dot(v2) / (v1.norm() * v2.norm())

    def _get_angle(self, name: str) -> Expr:
        "a·b=|a||b|cos θ:∠ABC"
        v1 = self._get_vec(name[1::-1])
        v2 = self._get_vec(name[1:])
        cos0 = self._get_vec_angle(v1,v2)
        return acos(cos0)

    def _get_n_vec(self, name: str) -> Expr:
        """
        a×b可作为平面法向量(sp)ABC
        大小为(向量围成的)平行四边形面积
        """
        v1 = self._get_vec(name[1::-1])
        v2 = self._get_vec(name[1:])
        return v1.cross(v2)
    def _get_triangle_area(self, name: str) -> Expr:
        "叉乘大小的一半可表示三角形面积"
        Sp = self._get_n_vec(name).norm()
        return Sp/Integer(2)

    def _get_plp_angle(self, plp: str) -> Expr:
        """
        二面角:A(p)-BC(l)-D(p)
        传入plp时'-'应省略,即(pang)ABCD
        cosθ=±cos<n1,n2>，异侧同角，同侧补角
        下面n1,n2为平面的异侧方向->θ=<n1,n2>
        """
        n1 = self._get_n_vec(plp[0:2])
        n2 = self.get_n_vec(plp[1:3])
        cos0 = self._get_vec_angle(n1,n2)
        return acos(cos0)

    def _get_ll_angle(self, l1: str, l2: str) -> Expr:
        """"
        (异面)直线(AB)与(CD)所成角(ll)
        传入ll时，应为(ll)AB/CD
        cosθ=|cos<v1,v2>|
        """
        v1 = self._get_vec(ll[0:1])
        v2 = self._get_vec(ll[2:4])
        cos0 = self._get_vec_angle(v1,v2)
        return acos(Abs(cos0))

    def _get_lp_angle(self, l: str, p: str) -> Expr:
        """
        直线AB与平面(sp)ABC所成角(lp)
        (lp)AB/spABC
        sinθ=|cos<v,n>|
        """
        v = self._get_vec(lp[0:1])
        n = self._get_n_vec(lp[5:7])
        cos0 = self._get_vec_angle(v,n)
        return asin(Abs(cos0))

    def _get_pp_angle(self, p1: str, p2: str) -> Expr:
        """
        平面(sp)ABC与平面(sp)BCD所成角(pp)
        (pp)spABC/spBCD
        cosθ=|cos<n1,n2>|
        """
        p1, p2 = tuple(pp.split("/"))
        n1 = self._get_n_vec(p1[2:4])
        n2 = self._get_n_vec(p2[2:4])
        cos0 = self._get_vec_angle(n1,n2)
        return acos(Abs(cos0))

    def _get_distance_pl(self, p: str, l: str) -> Expr:
        """
        点A到直线BC(底a)的距离(d)
        Sp=d*a=|va×vb|
        传入:dAtBC
        """
        Sp = self._get_n_vec(p+l).norm()
        a = self._get_distance(l)
        return Sp/a

    def _get_distance_pp(self, point: str, plane: str) -> Expr:
        """
        点A到平面BCD的距离(h1)
        点A到BC距离(h2)
        平面ABC与平面BCD所成角(θ)
        h1=h2*cosθ
        传入:dAtBCD
        """
        l = plane[0:1]
        d = self._get_distance_pl(point, l)
        sp1 = 'sp'+point+l
        sp2 = 'sp'+plane
        o = self._get_pp_angle(sp1+'/'+sp2)
        return d*cos(o)

    def _get_volume(self, trip: str) -> Expr:
        """
        三棱锥(四面体)A-BCD体积
        传入trip时应省略'-',即(v)ABCD
        V=1/3*d*S
        """
        point = trip[0]
        plane = trip[1:3]
        d = self._get_distance_pp(point,plane)
        S = self._get_triangle_area(plane)
        return d*S/Integer(3)

    def _eval_str_expr(self, expr: str) -> Expr | Never:
        """
        尝试解析字符串表达式，解析失败会报错
        别听 IDE 瞎说，这不是静态方法，``self`` 在 ``eval`` 里要用的
        """
        expr = mark_vec_coord(expr)
        rules = [
            # 幂运算符
            (r'\^', '**'),
            # 角度制
            ('deg', '* pi / 180'),
            # 给整数套上 ``Integer()``，防止一除变成小数
            (r'(?<!\.)\b(\d+)\b(?!\.)', r'Integer(\1)'),
            # 向量点乘
            ('dot', '@ dot @'),
            # 三维向量叉乘
            ('cross', '@ cross @'),
            # 未知数（不考虑排除 x, y 了，反正最后会报错）
            (r'\b([a-z]|' + '|'.join(VALID_GREEK_SPELLINGS) + r')\b', r"self._get_sp_symbol('\1')"),
            # 访问点坐标
            (r'\b(x|y)([A-Z])\b', r"self._get_\1_of('\2')"),
            # 线段长度
            (r'\b([A-Z]{2})\b', r"self._get_distance('\1')"),
            # 角度
            (r'\bang([A-Z]{3})\b', r"self._get_angle('\1')"),  # bang! 我这奇妙的笑点 233
            # 二面角
            (r'\bpang([A-Z]{4})\b', r"self._get_plp_angle('\1')"),  # pang! 我这奇妙的笑点 233
            # ll
            (r'\bll([A-Z]{2}/[A-Z]{2})\b', r"self._get_ll_angle('\1')"),
            # lp
            (r'\blp([A-Z]{2}/sp[A-Z]+)\b', r"self._get_lp_angle('\1')"),
            # pp
            (r'\bpp(sp[A-Z]+/sp[A-Z]+)\b', r"self._get_pp_angle('\1')"),
            # 点到平面的距离
            (r'\bd([A-Z])t([A-Z]{3})\b', r"self._get_distance_from_point_to_line('\1', '\2')"),
            # 三棱锥体积
            (r'\bv([A-Z]{4})\b', r"self._get_pp_angle('\1')"),
            # 平面(法向量)
            (r'\bsp([A-Z]{3})\b', r"self._get_n_vec('\1')"),
            # 两个大写字母的向量
            (r'\bvec([A-Z]{2})\b', r"self._get_vec('\1')"),
            # 三角形面积
            (r'\bSt([A-Z]{3})\b', r"self._get_triangle_area('\1')"),
            # 点到直线的距离
            (r'\bd([A-Z])t([A-Z]{2})\b', r"self._get_distance_pl('\1', '\2')")
        ]
        for pattern, repl in rules:
            expr = re.sub(pattern, repl, expr)
        return simplify(eval(expr))  # 不能用 ``sympy.sympify``，不然碰到没有的符号它会自己造

    def add_symbol(self, name: str, domain_settings: Optional[DomainSettings] = None):
        self._add_math_obj(GCSymbol_3d(name, domain_settings))
        self.symbol_names.append(name)

    def add_point(self, name: str, x_str: str, y_str: str, z_str: str, line1: str, line2: str, line3: str) -> None:
        """
        尝试添加点，并相应地添加依赖关系
        3d前端会发来 6 个字符串，其中 3 个是有内容的
        :param name: 点名称
        :param x_str: 横坐标的字符串表达式，若为 x 则设未知数
        :param y_str: 纵坐标的字符串表达式，若为 y 则设未知数
        :param z_str: 纵坐标的字符串表达式，若为 z 则设未知数,留空回归平面计算(z=0)
        :param line1: 该点所在的直线 1
        :param line2: 该点所在的直线 2
        :param line3: 该点所在的直线 3
        """
        try:
            eqs: list[Eq] = []
            required_by_new_symbols: set[str] = set()

            # 设未知数
            if x_str == 'x':
                self.add_symbol(f'x_{name}')
            if y_str == 'y':
                self.add_symbol(f'y_{name}')
            if z_str == 'z':
                self.add_symbol(f'z_{name}')

            # 先设完未知数再读取处理，防止干扰依赖关系
            if x_str != '':
                if x_str == 'x':
                    eqs.append(Eq(x, self._get_sp_symbol(f'x_{name}')))
                    required_by_new_symbols.add(f'x_{name}')
                else:
                    eqs.append(Eq(x, self._eval_str_expr(x_str)))
            if y_str != '':
                if y_str == 'y':
                    eqs.append(Eq(y, self._get_sp_symbol(f'y_{name}')))
                    required_by_new_symbols.add(f'y_{name}')
                else:
                    eqs.append(Eq(y, self._eval_str_expr(y_str)))
            if len(z_str)==0:
                eqs.append(Eq(z, Integer(0))) #平面模式'
            else:
                if z_str == 'z':
                    eqs.append(Eq(z, self._get_sp_symbol(f'z_{name}')))
                    required_by_new_symbols.add(f'z_{name}')
                else:
                    eqs.append(Eq(z, self._eval_str_expr(z_str)))

            for l in [line1, line2]:
                if l != '':
                    eqs.append(self._get_line(l).equation())

            # 求解点坐标并添加
            solution = solve(eqs, x, y, z, dict=True)[0]
            point = GCPoint_3d(name, [solution[x],solution[y],solutuon[z]])
            # 反向添加设的未知数对点的依赖，这样在删除点时该点的未知数也会被删除
            point.required_by |= required_by_new_symbols
            self._add_math_obj(point)
            self.point_names.append(name)

        except Exception as e:
            # 清理可能添加的未知数
            for n_i in "xyz":
                if f'{n_i}_{name}' in self.symbol_names:
                    self.symbol_names.remove(name)
                    del self.math_objs[name]
            self.requirements_tracker.clear()
            raise e

    @AddBinCond_3d('=')
    def add_expr_eq(self, input1: str, input2: str) -> list[Eq]:
        """两表达式相等"""
        return [Eq(self._eval_str_expr(input1), self._eval_str_expr(input2))]

    @AddBinCond_3d(r'\parallel')
    def add_parallel(self, input1: str, input2: str) -> list[Eq]:
        "两直线平行|a×b|=0"
        v1=self._get_vec(input1)
        v2=self._get_vec(input2)
        n=v1 @ cross @ v2
        return [Eq(n.norm(), Integer(0))]

    @AddBinCond_3d(r'\perp')
    def add_perp(self, input1: str, input2: str) -> list[Eq]:
        "两直线垂直a·b=0"
        v1=self._get_vec(input1)
        v2=self._get_vec(input2)
        return [Eq(v1 @ dot @ v2, Integer(0))]

    @AddBinCond_3d(r'\cong')
    def add_cong(self, input1: str, input2: str) -> list[Eq]:
        """三角形全等（SSS）"""
        a1, b1, c1 = input1[:2], input1[1:], input1[0] + input1[2]
        a2, b2, c2 = input2[:2], input2[1:], input2[0] + input2[2]
        eqs = []
        for s1, s2 in [(a1, a2), (b1, b2), (c1, c2)]:
            eqs.append(Eq(self._get_distance(s1), self._get_distance(s2)))
        return eqs

    @AddBinCond_3d(r'\sim')
    def add_sim(self, input1: str, input2: str) -> list[Eq]:
        """三角形相似 (SSS)"""
        a1, b1, c1 = input1[:2], input1[1:], input1[0] + input1[2]
        a2, b2, c2 = input2[:2], input2[1:], input2[0] + input2[2]
        k1 = self._get_distance(a1) / self._get_distance(a2)
        k2 = self._get_distance(b1) / self._get_distance(b2)
        k3 = self._get_distance(c1) / self._get_distance(c2)
        return [Eq(k1, k2), Eq(k2, k3)]

    @AddUnaryCond_3d('平行四边形')
    def add_parallelogram(self, input1: str) -> list[Eq]:
        v1 = self._get_vec(input1[:2])
        v2 = self._get_vec(input1[:1:-1])
        return [Eq(v1, v2)]

    @AddUnaryCond_3d('菱形')
    def add_rhombus(self, input1: str) -> list[Eq]:
        opposite1, opposite2 = input1[:2], input1[:1:-1]
        adjacent = input1[1:3]
        return [
            Eq(self._get_vec(opposite1), self._get_vec(opposite2)),
            Eq(self._get_distance(opposite1), self._get_distance(adjacent))
        ]

    @AddUnaryCond_3d('矩形')
    def add_rect(self, input1: str) -> list[Eq]:
        opposite1, opposite2 = input1[:2], input1[:1:-1]
        adjacent = input1[1:3]
        return [
            Eq(self._get_vec(opposite1), self._get_vec(opposite2)),
            Eq(self._get_vec(opposite1) @ dot @ self._get_vec(adjacent), 0)
        ]

    @AddUnaryCond_3d('正方形')
    def add_square(self, input1: str) -> list[Eq]:
        opposite1, opposite2 = input1[:2], input1[:1:-1]
        adjacent = input1[1:3]
        return [
            Eq(self._get_vec(opposite1), self._get_vec(opposite2)),
            Eq(self._get_distance(opposite1), self._get_distance(adjacent)),
            Eq(self._get_vec(opposite1) @ dot @ self._get_vec(adjacent), 0)
        ]

    @AddUnaryCond_3d('等边三角形')
    def add_equilateral_triangle(self, input1: str) -> list[Eq]:
        s1 = self._get_distance(input1[:2])
        s2 = self._get_distance(input1[1:])
        s3 = self._get_distance(input1[0] + input1[2])
        return [Eq(s1, s2), Eq(s2, s3)]

    def get_symbol_names(self) -> list[str]:
        return self.symbol_names

    def get_point_names(self) -> list[str]:
        return self.point_names

    def get_cond_ids(self) -> list[str]:
        return self.cond_ids

    def get_symbols_latex(self) -> list[LatexItem]:
        """
        获取需要在前端页面上展示的符号的 LaTeX，包含取值范围（含始末 $ $）
        相同取值范围的符号会被并到一起
        :return: 一个列表，每项为一个字典（对象）
                 id: 取值范围的 LaTeX，用于前端 ``v-for`` 的 ``key``
                 latex: 该取值范围的完整的 LaTeX
        """
        # 将每个符号名挂到其取值范围上
        domain_names_dict: dict[str, list[str]] = {}
        for name in self.symbol_names:
            gc_symbol: GCSymbol_3d = self.math_objs[name]  # type: ignore
            name_latex = gc_symbol.get_name_latex()
            domain_latex = gc_symbol.get_domain_latex()
            if domain_latex not in domain_names_dict:
                domain_names_dict[domain_latex] = []
            domain_names_dict[domain_latex].append(name_latex)

        # 生成结果
        result = []
        for domain, names in domain_names_dict.items():
            result.append({
                'id': domain,
                'latex': fr"$ \displaystyle {', '.join(names)} \in {domain} $"
            })

        return result

    def get_points_latex(self) -> list[LatexItem]:
        """获取所有点的 LaTeX（含始末 $ $）"""
        result = []
        for name in self.point_names:
            result.append({
                'id': name,
                'latex': fr'$ \displaystyle {self.math_objs[name].get_latex()} $'  # type: ignore
            })
        return result

    def get_conds_latex(self) -> list[LatexItem]:
        """获取所有条件的 LaTeX，包括原始的和方程的（均含始末 $ $）"""
        result = []
        for cond_id in self.cond_ids:
            cond: Cond = self.math_objs[cond_id]  # type: ignore
            result.append({
                'id': fr'$$ {cond.get_raw_latex()} $$',
                'latex': cond.get_eqs_latex()
            })
        return result

    def get_deeply_required_by(self, identifier: str) -> list[str]:
        """
        查询一个对象被哪些对象依赖（包括其后代的依赖）
        :param identifier: 需要查询的对象的 ``id``
        :return: 一个列表（实际上是一个集合），所有被依赖的对象的 ``id``
        """
        # BFS
        result = set()
        visited = {identifier}
        queue = deque([identifier])

        while len(queue) > 0:
            current_id = queue.popleft()
            for i in self.math_objs[current_id].required_by:
                if i not in visited:
                    result.add(i)
                    visited.add(i)
                    queue.append(i)

        return list(result)

    def del_objs(self, ids: list[str]) -> None:
        for i in ids:
            # 删除对象
            del self.math_objs[i]
            # 列表除名
            for l in [self.symbol_names, self.point_names, self.cond_ids]:
                if i in l:
                    l.remove(i)
        # 删除依赖关系
        for obj in self.math_objs.values():
            obj.required_by -= set(ids)

    def save_to_file(self) -> None:
        path = windows[0].create_file_dialog(SAVE_DIALOG, file_types=('几何计算器 pickle 文件 (*_3d.gc.pkl)',))
        if path is not None:
            # path = path[0]
            # # https://github.com/r0x0r/pywebview/issues/1677
            with open(path, 'wb') as f:
                pickle.dump(self, f)

    def load_from_file(self) -> None:
        path = windows[0].create_file_dialog(OPEN_DIALOG, file_types=('几何计算器 pickle 文件 (*_3d.gc.pkl)',))
        if path is not None:
            path = path[0]
            with open(path, 'rb') as f:
                self.__dict__ = pickle.load(f).__dict__

    def solve(self, expr: str) -> list[str]:
        """
        🚀 启动！
        :param expr: 要求解的目标的字符串表达式
        :return: 所有可能的解的 LaTeX
        """
        left = to_raw_latex_3d(expr)

        target = Symbol('target')
        eqs = [Eq(target, self._eval_str_expr(expr))]
        for i in self.cond_ids:
            eqs.extend(self.math_objs[i].eqs)  # type: ignore
        symbols = [target] + [self.math_objs[i].sp_symbol for i in self.symbol_names]  # type: ignore
        solutions = solve(eqs, symbols, dict=True)

        # 关于 ``sqrtdenest``：https://github.com/zhdbk3/GeometryCalculator/issues/5
        result = set(simplify(sqrtdenest(s[target])) for s in solutions)
        result = [f'{left} = {latex(i)}' for i in result]
        return result
