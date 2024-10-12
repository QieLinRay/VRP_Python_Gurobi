"""
Created By: Miaomiao Wang SHU
Date: 2023 07 17
Problem: VRP
Tool: Gurobi
Notions: 学习Py+Gurobi+VRP的代码
"""
import gurobipy as gb
import numpy as np
import pandas as pd
import random
import time

# 问题规模
N = 10
K = 5

# 参数设置，每个客户点的坐标
seed = 10
random.seed(seed)
x_i = [random.randint(0, 20) for i in range(N + 2)]
y_i = [random.randint(0, 20) for i in range(N + 2)]
x_i[N + 1] = x_i[0]
y_i[N + 1] = y_i[0]
M = 2 * N  # 为什么这里M取2N就够了呢
x1 = pd.Series(x_i)  # pandas.Series()是pandas库中的一个函数，用于创建一个一维的带标签的数组
y1 = pd.Series(y_i)
out_dij = pd.DataFrame({'x_i': x1, 'y_i': y1})  # 创建二维表格：通过传递一个字典、嵌套序列或其他数据结构，可以创建一个带有行和列的二维表格
print(out_dij)
d_ij = np.ones((N + 2, N + 2), dtype=int)  # numpy.ones()是用于创建一个指定形状的数组，并将数组中的元素全部初始化为1的函数
for i in range(N + 2):
    for j in range(N + 2):
        d_ij[i][j] = abs(x_i[i] - x_i[j]) + abs(y_i[i] - y_i[j])
for i in range(N + 2):
    for j in range(N + 2):
        print('{:< 4}'.format(d_ij[i][j]),end=" ")
    print()
# '{:<4}' 是一个格式化字符串，表示输出一个长度为4的左对齐字符串，字符串的内容由后面的变量 d_ij[i][j] 提供。
# 具体来说，{} 中的 < 表示左对齐，4 表示总宽度为 4，如果 d_ij[i][j] 的长度小于 4，那么会在右侧补齐空格，如果大于 4，则不会进行截断。


def gurobi_model():
    # 定义模型
    model = gb.Model("VRP")
    model.setParam("Timelimit", 600)
    model.setParam('Method', 3)
    model.setParam('MIPGap', 0)

    # variables
    alpha_ki = {}
    beta_kij = {}
    omiga_ki = {}
    for k in range(K):
        for i in range(N + 2):
            name1 = "alpha_{0}_{1}".format(k, i)
            name2 = "omiga_{0}_{1}".format(k, i)
            alpha_ki[k, i] = model.addVar(0, 1, vtype=gb.GRB.BINARY, name=name1)
            omiga_ki[k, i] = model.addVar(0, N + 5, vtype=gb.GRB.CONTINUOUS, name=name2)
            for j in range(N + 2):
                name3 = "beta_{0}_{1}_{2}".format(k, i, j)
                if i != j:
                    beta_kij[k, i, j] = model.addVar(0, 1, vtype=gb.GRB.BINARY, name=name3)
                else:
                    beta_kij[k, i, j] = model.addVar(0, 0, vtype=gb.GRB.BINARY, name=name3)
    model.update()

    obj = gb.LinExpr(0)
    for k in range(K):
        for i in range(N + 2):
            for j in range(N + 2):
                obj.addTerms(d_ij[i][j], beta_kij[k, i, j])
    model.setObjective(obj, gb.GRB.MINIMIZE)

    for i in range(1, N+1):  # 注意不是N+2，头尾节点不需要
        expr1 = gb.LinExpr(0)
        for k in range(K):
            expr1.addTerms(1, alpha_ki[k, i])
        model.addConstr(expr1 == 1, name="service assigned for{}".format(i))

    # 一头一尾
    for k in range(K):
        expr21 = gb.LinExpr(0)
        expr22 = gb.LinExpr(0)
        for i in range(0, N + 1):
            expr21.addTerms(1, beta_kij[k, i, N + 1])
        model.addConstr(expr21 == 1, name="end_point for{0}".format(k))
        for i in range(1, N + 2):
            expr22.addTerms(1, beta_kij[k, 0, i])
        model.addConstr(expr22 == 1, name="start_point for{0}".format(k))

    # 一前一后
    for k in range(K):
        for i in range(N):
            expr31 = gb.LinExpr(0)
            expr32 = gb.LinExpr(0)
            for j in range(0, N + 1):
                expr31.addTerms(1, beta_kij[k, j, i])
            model.addConstr(expr31 == alpha_ki[k, i], name="front_point_k{0}_i{1}".format(k, i))
            for j in range(1, N + 2):
                expr32.addTerms(1, beta_kij[k, i, j])
            model.addConstr(expr32 == alpha_ki[k, i],name="front_point_k{0}_i{1}".format(k, i))

    # 去子路
    for k in range(K):
        for i in range(0, N + 1):
            for j in range(1, N + 2):
                model.addConstr(omiga_ki[k, j] >= omiga_ki[k, i] - M * (1 - beta_kij[k, i, j]) + 1,
                                name="subroad_{0}_{1}_{2}".format(k, i, j))

    model.write('gurobi_VRP.LP')

    model.optimize()
    if model.status == gb.GRB.Status.OPTIMAL or model.status == gb.GRB.Status.TIME_LIMIT:
        print("obj = {0}".format(model.ObjVal))
    return model.ObjVal
if __name__=="__main__":
    print("-----VRP_Gurobi------")
    start_time2 = time.time()  # 记录模型开始时间
    result_gurobi = gurobi_model()
    end_time2 = time.time()
    print("gurobi_model 运行结果={0}  求解时间={1:<.3} 秒".format(result_gurobi, end_time2 - start_time2))



