from logger import frontend_logger
from problem import Problem
from problem_3d import Problem_3d

class API:
    problem = Problem()
    logger = frontend_logger
    problem_3d = Problem_3d()

api = API()
