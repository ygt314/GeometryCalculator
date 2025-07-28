from functools import cache
from sympy import Expr, latex, Point
from .math_obj import MathObj
"""
Point:自动挡的点对象
Point(1,2)->Point2D(1,2)
Point(1,2,3)->Point3D(1,2,3)
支持加减:Point(1,2)-Point(2,3)=Point2D(-1,-1)
支持化矩阵:Matrix(Point(1,2))=Matrix([1,2])
"""
class GCPoint(MathObj):
    def __init__(self, name: str, xy: list[Expr, Expr]):
        """
        几何计算器中的点
        :param name: 点名称，一个大写字母
        :param x: 横坐标
        :param y: 纵坐标
        """
        super().__init__(name)
        self.x, self.y= tuple(xy)
        self.sp_point = Point(xy)
    @cache
    def get_latex(self) -> str:
        return fr'{self.id} \left( {latex(self.x)}, {latex(self.y)} \right)'
class GCPoint_3d(MathObj):
    def __init__(self, name: str, xyz: list[Expr, Expr, Expr]):
        """
        3d几何计算器中的点
        :param name: 点名称，一个大写字母
        :param x: 横坐标
        :param y: 纵坐标
        :param z: 竖坐标
        """
        super().__init__(name)
        self.x, self.y, self.z= tuple(xyz)
        self.sp_point = Point(xyz)
    @cache
    def get_latex_3d(self) -> str:
        return fr'{self.id} \left( {latex(self.x)}, {latex(self.y)}, {latex(self.z)} \right)'
