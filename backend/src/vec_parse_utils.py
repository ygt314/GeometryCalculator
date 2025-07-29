from typing import Callable


def mark_vec_coord(expr: str) -> str:
    """
    用 ``Matrix([])`` 标记表达式中向量的坐标表示
    现支持多维向量
    """
    pattern = r'(?<!Matrix)\(([\d,],[\d,]+)\)'
    result = re.sub(pattern,r'Matrix([\1])',expr)
    return result


class Infix:
    """【Python 竟然允许这种语法， Python中缀运算符】 https://www.bilibili.com/video/BV1Xe411r7VE
    用法: A @ dot(cross) @ B"""

    def __init__(self, func: Callable):
        self.func = func

    def __rmatmul__(self, other) -> 'Infix':
        return Infix(lambda var: self.func(other, var))

    def __matmul__(self, other):
        return self.func(other)


dot = Infix(lambda a, b: a.dot(b))
cross = Infix(lambda a, b: a.cross(b))
