import gurobipy as grb
import data
import time


def CVRP():
    """
    Use gurobi to solve CVRP
    :return:model
    """
    model = grb.Model("CVRP")
    model.setParam("TimeLimit", 3600)
    model.setParam("Method", 1)  # Simplex
    model.setParam("outputFlag", 1)

    # add variables
    gamma_nk = {}
    alpha_nk = {}
    beta_nn1k = {}
    # gamma_nk
    for n in range(data.N + 2):
        for k in range(data.K):
            name1 = "gamma_nk[{0}][{1}]".format(n,k)
            gamma_nk[n,k] = model.addVar(0,grb.GRB.INFINITY,vtype=grb.GRB.CONTINUOUS,name=name1)
    # alpha_nk
    for n in range(data.N + 2):
        for k in range(data.K):
            name1 = "alpha_nk[{0}][{1}]".format(n,k)
            alpha_nk[n,k] = model.addVar(0,1,vtype=grb.GRB.BINARY,name=name1)
    # beta_nn1k
    for n in range(data.N + 2):
        for n1 in range(data.N + 2):
            for k in range(data.K):
                name1 = "beta_nn1k[{0}][{1}][{2}]".format(n,n1,k)
                beta_nn1k[n,n1,k] = model.addVar(0,1,vtype=grb.GRB.BINARY,name=name1)

    # add constraints
    for n in range(1,data.N + 1):
        expr1 = grb.LinExpr(0)
        for k in range(data.K):
            expr1.addTerms(1,alpha_nk[n,k])
        model.addConstr(expr1 == 1, name="Constr1")

    for n1 in range(1,data.N + 1):
        for k in range(data.K):
            expr2_1 = grb.LinExpr(0)
            expr2_2 = grb.LinExpr(0)
            for n in range(1, data.N + 2):
                expr2_1.addTerms(1,beta_nn1k[n1,n,k])
            for n in range(0,data.N + 1):
                expr2_2.addTerms(1,beta_nn1k[n,n1,k])
            model.addConstr(expr2_1 == expr2_2,name="Constr2_1")
            model.addConstr(expr2_1 == alpha_nk[n1,k],name="Constr2_2")

    for k in range(data.K):
        expr3_1 = grb.LinExpr(0)
        expr3_2 = grb.LinExpr(0)
        for n in range(0,data.N + 1):
            expr3_1.addTerms(1,beta_nn1k[n,data.N + 1,k])
        for n in range(1,data.N + 2):
            expr3_2.addTerms(1,beta_nn1k[0,n,k])
        model.addConstr(expr3_1 == expr3_1,name="Constr3_1")
        model.addConstr(expr3_1 ==  1,name="Constr3_2")

    for n in range(1,data.N + 1):
        for n1 in range(0,data.N):
            for k in range(data.K):
                model.addConstr(gamma_nk[n,k] >= gamma_nk[n1,k] + data.t_nn1[n][n1] -
                                data.M *(1-beta_nn1k[n1,n,k]),name="Constr4")

    for k in range(data.K):
        expr5 = grb.LinExpr(0)
        for n in range(1, data.N):
            expr5.addTerms(data.q_n[n],alpha_nk[n,k])

    # OBJ
    obj = grb.LinExpr(0)
    for k in range(data.K):
        for n in range(0,data.N):
            for n1 in range(1,data.N + 1):
                obj.addTerms(data.t_nn1[n][n1], beta_nn1k[n,n1,k])
    model.setObjective(obj,grb.GRB.MAXIMIZE)

    model.optimize()
    if model.status == grb.GRB.Status.INFEASIBLE:
        print("CVRP IS INFEASIBLE")
        model.computeIIS()
        model.write("CVRP + {0}.ilp".format(time.time()))
        return model
    elif model.status == grb.GRB.TIME_LIMIT:
        print("RUN OUT OF TIME LIMIT")
        return model
    elif model.status == grb.GRB.OPTIMAL:
        print("solveByGurobi_obj = {0}".format(model.ObjVal))

        def printVars():
            """
            print the optimal solution
            :return:
            """
            for n in range(1,data.N + 1):
                for k in range(data.K):
                    if gamma_nk[n,k].x > 0:
                        print("gamma_nk[{0}][{1}] = {2}".format(n,k,gamma_nk[n,k].x))

            for n in range(1,data.N + 1):
                for k in range(data.K):
                    if alpha_nk[n,k].x > 0.9:
                        print("alpha_nk[{0}][{1}] = {2}".format(n,k,alpha_nk[n,k].x))

            for n in range(data.N + 2):
                for n1 in range(data.N + 2):
                    for k in range(data.K):
                        if beta_nn1k[n,n1,k].x > 0.9 :
                            print("beta_nn1k[{0}][{1}][{2}] = {3}".format(n,n1,k,beta_nn1k[n,n1,k].x))

        printVars()
        return model






