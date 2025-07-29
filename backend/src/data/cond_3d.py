import re

from sympy import Eq, latex, sympify
from sympy.printing.latex import LatexPrinter

from .math_obj import MathObj
from vec_parse_utils import mark_vec_coord

def map_vec_coord_3d(expr: str) -> tuple[str, dict[str, str]]:
    """
    将向量的坐标表示（已标记为 ``Matrix``）映射到一个变量上，以便 ``sympy.sympify`` 将它直接当作一个普通符号解析
    （这么做是因为即使设置了 ``sympy.sympify(evaluate=False)`` 也无法阻止向量“数乘”的执行）
    :param expr: 原始字符串表达式
    :return: 两个值：
             - 替换后的字符串表达式，可以直接喂给 ``sympy.sympify`` 解析
             - 映射表，键为向量坐标映射到的变量名，值为它自己的 LaTeX
    """
    mapping = {}
    while True:
        left_index = expr.find('Matrix([')
        if left_index != -1:
            right_index = expr.find('])') + 2
            vec_coord = expr[left_index:right_index]
            # 用哈希值确保映射的唯一性，平方消除负号
            # 末尾有字母，数字不会被 ``sympy.sympify` 变成下标
            alias = f'vec{hash(vec_coord) ** 2}coord'
            expr = expr.replace(vec_coord, alias)
            mapping[alias] = latex(sympify(vec_coord, evaluate=False), mul_symbol='dot')
        else:
            return expr, mapping


def to_raw_latex_3d(expr: str) -> str:
    """
    生成出用户原始输入的表达式的 LaTeX
    只能是单个表达式
    """
    expr = (mark_vec_coord(expr)
            .replace('deg', '* gcdeg')  # SymPy 内有个函数就叫 ``deg``，故在此做区分
            .replace('dot', '*')
            .replace('cross','×')) # ×表示叉乘

    expr, mapping = map_vec_coord(expr)

    expr = latex(sympify(expr, evaluate=False), mul_symbol='dot')

    for alias, vec_coord_latex in mapping.items():
        expr = expr.replace(alias, vec_coord_latex)

    rules = [
        # ·gcdeg -> °
        (r'\\cdot\s+gcdeg', r'^{\\circ}'),
        #          ->
        # vecAB -> AB
        (r'\bvec([A-Z]{2})\b', r'\\overrightarrow{\1}'),
        # trABC -> △ABC,区分平面表示
        (r'\btr([A-Z]{3})\b', r'\\triangle \1'),
        # angABC -> ∠ABC
        (r'\bang([A-Z]{3})\b', r'\\angle \1'),
        # 删除多余点号
        (r'(?<=[0-9a-z])\s*\\cdot\s*(?=[a-zA-Z]|\\overrightarrow)', r' '),
        # pangABCD -> ∠A-BC-D
        (r'\bpang([A-Z])([A-Z]{2})([A-Z])\b', r'\\angle \1-\2-\3'),
        # llAB/CD -> ∠AB-CD
        (r'\bll([A-Z]{2})/([A-Z]{2})\b', r"\\angle \1-\2"),
        # lpAB/spBCD -> ∠AB-spBCD
        (r'\blp([A-Z]{2})/(sp[A-Z]+)\b', r"\\angle \1-\2"),
        # ppspABC/spBCD -> ∠spABC-spBCD
        (r'\bpp(sp[A-Z]+)/(sp[A-Z]+)\b', r"\\angle \1-\2"),
        # spABC -> 平面ABC(法向量)
        (r'\bsp([A-Z]+\b)',r'平面\1'),
        # dAtBCD -> d_{A 到 平面BCD}
        (r'\bd([A-Z])t([A-Z]{3})\b', r'd_{\1 到 平面\2}'),
        # vABCD -> V_四面体ABCD
        (r'\bv([A-Z]{4})\b', r'V_{四面体\1}'),
        # StABC -> S△ABC
        (r'\bSt([A-Z]{3})\b', r'S_{\\triangle \1}'),
        # xA -> x_A
        (r'\b(x|y)([A-Z])\b', r'\1_\2'),
        # dAtBC -> d_{A 到 BC}
        (r'\bd([A-Z])t([A-Z]{2})\b', r'd_{\1 到 \2}')
    ]
    for pattern, repl in rules:
        expr = re.sub(pattern, repl, expr)

    return expr


class CustomLatexPrinter_3d(LatexPrinter):
    def _print_MatrixBase(self, expr):
        return self._print_Tuple((*expr,))


class Cond_3d(MathObj):
    def __init__(self, raw_latex: str, eqs: list[Eq]):
        """
        一个条件，可以时简单的，也可以是复合的
        :param raw_latex: 用户原始输入的 LaTeX 形式
        :param eqs: 解析得到的方程，可能需要多个
        """
        super().__init__(raw_latex)
        self.eqs = eqs

    def get_raw_latex(self) -> str:
        # 前面用原始 LaTeX 做了 ``id``
        return self.id

    def get_eqs_latex(self) -> str:
        return ' '.join(f'$$ {latex(eq)} $$' for eq in self.eqs)
