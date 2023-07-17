# VRP_Python_Gurobi
使用python调用Gurobi解决VRP问题
## 20230717 VRP_Gurobi_Python总结

### 0.本文框架

1. **python调用gurobi框架**：首先给出python调用gurobi的大框架，总体上从模型声明，变量声明，约束，目标函数的构建，模型求解的大方向进行介绍

2. **代码总结**：将对框架中的内容中自身觉得重要的地方进行详细介绍

3. **详细代码展示**：将详细展示使用python调用gurobi求解VRP模型的代码。

### 1.**python调用gurobi框架**

该部分给出在建模过程中的代码调用框架。用python调用gurobi求解模型的时候，步骤主要由以下6步骤构成。以下给出详细的框架

```python
# 0 import API
import gurubipy as gb
# 1设置参数和集合
# 根据自身问题集合参数

# 2声明模型并设置模型求解参数
model = gb.Model("ModelName")
model.setParam("Timelimit", 600)

# 3声明决策变量：以10x10的决策变量为例
x = {}
for i in range(10):
	for j in range(10):
        name = "x_{0}_{1}".format(i,j)
        model.addVar(lb,up,vtype=,name=name1)
# 3.1更新模型
model.update()

# 4创建目标函数
# expr = gb.LinExpr(0) :创建线性表达式

obj = gb.LinExpr(0)
for i in range():
    obj.addTerms(系数，变量)
model.setObjective(obj, gb.GRB.MINIMIZE)

# 5 创建约束
for i in range():
    expr = gb.LinExpr(0)
    for j in range():
        expr.addTerms(系数，变量)
  	model.addConstr(expr>=1,name=xx)


# 6.0 将模型写入LP文件
model.write("name.Lp")
# 6.1 模型求解
model.optimize()

```

### **2.代码总结**

**2.1总结**

```python
# 在求解过程中常用的API
import gurobipy as gb  
import numpy as np
import pandas as pd
import random
import time
```

**2.1.1模型初始化**

```python
model = gb.Model("VRP")  # 模型初始化
model.setParam("Timelimit", 600) 
model.setParam('Method', 3)
model.setParam('MIPGap', 0)
```

在Gurobi中，`model.setParam()`方法是用来设置Gurobi模型的参数的。该方法的语法如下：

```python
model.setParam(paramname, newvalue)  # `paramname`是要设置的参数名称，`newvalue`是新的参数值
```

下面给出常用的参数列表：

`TimeLimit`: 求解器的最大运行时间（单位为秒）。默认值为`inf`，表示没有时间限制。

`MIPGap`: 求解器的最优性容忍度（MIP gap）

在Gurobi中，`Method`参数用于设置线性规划求解器的求解方法。该参数可以设置为以下几个值之一：

- `0`（默认值）：自动选择求解方法，根据问题的特性选择最佳求解方法。
- `1`：使用单纯形法求解线性规划问题。
- `2`：使用内点法求解线性规划问题。
- `3`：使用双红线法求解线性规划问题。

`MIPFocus`: 求解器的求解重点。可以设置为`0`（默认值，表示平衡）、`1`（强调发现可行解）、`2`（强调发现最优解）或`3`（强调发现证明最优解）。

`Threads`: 求解器使用的线程数。默认值为`1`。

`OutputFlag`: 是否输出求解器的详细信息。默认值为`1`（输出），可以设置为`0`（不输出）。

```python
model.setParam('TimeLimit', 3600)  # 设置求解器的最大运行时间为1小时
model.setParam('MIPGap', 0.01)  # 设置求解器的最优性容忍度为1%
model.setParam('MIPFocus', 2)  # 设置求解器的求解重点为发现最优解
model.setParam('Threads', 4)  # 设置求解器使用4个线程
model.setParam('OutputFlag', 0)  # 关闭求解器的输出
```



**2.1.2变量声明**

变量声明的套路。

```python
a = {}
for i in range():
    for j in range():
        ...
        name1 = "name{0}{1}..".format(i, j ...)
        a[i,j,...] = model.addVar(lb,up,vtype= GRB.XX, name = name1)
```

声明k×i维的二维0-1决策变量,需要注意的是，这里是addVar(),而不是addVars(),不然会报错

```python
 alpha_ki = {}
    for k in range(K):
        for i in range(N + 2):
            name1 = "alpha_{0}_{1}".format(k, i)
            alpha_ki[k, i] = model.addVar(0, 1, vtype=gb.GRB.BINARY, name=name1)
```

声明i != j时，为1的0-1决策变量，需要在决策变量中声明的时候使用if else判断,而不是在后面加上约束的时候再判断

```python
beta_kij = {}
  for k in range(K):
    for i in range(N + 2):
        for j in range(N + 2):
                name3 = "beta_{0}_{1}_{2}".format(k, i, j)
                if i != j:
                    beta_kij[k, i, j] = model.addVar(0, 1, vtype=gb.GRB.BINARY, name=name3)
                else:
                    beta_kij[k, i, j] = model.addVar(0, 0, vtype=gb.GRB.BINARY, name=name3)
```

**2.1.3模型更新**

在Gurobi中，`model.update()`方法是用来更新模型的约束和变量的。在Python中，我们通常是先创建模型对象，然后添加约束和变量，最后调用`model.optimize()`方法求解模型。在这个过程中，我们可以通过调用`model.update()`方法来更新模型。

具体而言，当我们向模型中添加新的约束或变量时，这些约束或变量的信息会被存储在Gurobi内部的数据结构中，而不会立即传输到Gurobi的求解器中。只有在调用`model.update()`方法后，Gurobi才会将这些约束和变量的信息传输到求解器中，使其可以被求解器使用。

因此，在调用`model.optimize()`方法之前，我们通常会先调用`model.update()`方法，以确保模型的约束和变量已经被传输到求解器中，从而可以正确地求解模型。

需要注意的是，当我们修改模型中的约束或变量时，也需要再次调用`model.update()`方法，以确保修改后的模型信息已经被传输到求解器中。

```python
model.update()
```

**2.1.4 目标函数**

```python
obj = gb.LinExpr(0) # 创建线性约束
for i in range():
	...
	obj.addTerms(系数，决策变量)
model.setObjective(obj,gb.GRB.MINIMIZE)
```

**2.1.5约束**

套路是先初始化LinExpr(),然后expr.addTerms(),然后用model.addConstr()添加约束

再gurobi中，约束可以直接写加减乘除，不用像cplex那么麻烦

```python
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
```

首先总结，在本次代码中学到的新语法

```python
# 设置随机种子
seed = 10
random.seed(seed)

# 生成长度为N+2的列表
x_i = [random.randint(0, 20) for i in range(N + 2)]

# 创建一个初始值为1的二维数组
d_ij = np.ones((N + 2, N + 2), dtype=int)

# pandas.Series()函数
#  pandas.Series()是pandas库中的一个函数，用于创建一个一维的带标签的数组
x1 = pd.Series(x_i)

# pandas.DataFrame()创建一个二维表格
out_dij = pd.DataFrame({'x_i': x1, 'y_i': y1})

```

**2.1.6将模型写入lp文件（可选）**

```python
model.write('gurobi_VRP.LP')
```



**2.1.7模型求解**

```python
model.optimize()
```



**2.1.8main函数调用**

```python
if __name__=="__main__":
    print("-----VRP_Gurobi------")
    start_time2 = time.time()  # 记录模型开始时间
    result_gurobi = gurobi_model()
    end_time2 = time.time()
    # print("cplex_model  运行结果={0}  求解时间={1:<.3} 秒".format(result_cplex,end_time - start_time))
    print("gurobi_model 运行结果={0}  求解时间={1:<.3} 秒".format(result_gurobi, end_time2 - start_time2))


```

### 3.**详细代码**

```python
"""
Created By: Miaomiao Wang SHU
Date: 2023 07 17
Problem: VRP
Tool: Gurobi
Notations: 自我学习代码
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
    # print("cplex_model  运行结果={0}  求解时间={1:<.3} 秒".format(result_cplex,end_time - start_time))
    print("gurobi_model 运行结果={0}  求解时间={1:<.3} 秒".format(result_gurobi, end_time2 - start_time2))


```




