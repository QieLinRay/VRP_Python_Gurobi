import gurobipy as grb
import data
import time


class CVRP:
    """
    Use gurobi to solve CVRP
    call solve() method to use this model
    """

    def __init__(self, data, time_limit=3600, method=1, output_flag=1, isprint=True):
        self.model = grb.Model("CVRP")
        self.model.setParam("TimeLimit", time_limit)
        self.model.setParam("Method", method)  # method1: Simplex
        self.model.setParam("outputFlag", output_flag)

        # whether to print answer
        self.isprint = isprint

        self.data = data

    def add_variables(self):

        # add variables
        self.gamma_nk = {}
        self.alpha_nk = {}
        self.beta_nn1k = {}

        # gamma_nk
        for n in range(self.data.N + 2):
            for k in range(self.data.K):
                name1 = "gamma_nk[{0}][{1}]".format(n, k)
                self.gamma_nk[n, k] = self.model.addVar(0, grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS, name=name1)
        # alpha_nk
        for n in range(self.data.N + 2):
            for k in range(self.data.K):
                name1 = "alpha_nk[{0}][{1}]".format(n, k)
                self.alpha_nk[n, k] = self.model.addVar(0, 1, vtype=grb.GRB.BINARY, name=name1)
        # beta_nn1k
        for n in range(self.data.N + 2):
            for n1 in range(self.data.N + 2):
                for k in range(self.data.K):
                    name1 = "beta_nn1k[{0}][{1}][{2}]".format(n, n1, k)
                    self.beta_nn1k[n, n1, k] = self.model.addVar(0, 1, vtype=grb.GRB.BINARY, name=name1)

    def add_constraints(self):
        # add constraints
        for n in range(1, self.data.N + 1):
            expr1 = grb.LinExpr(0)
            for k in range(self.data.K):
                expr1.addTerms(1, self.alpha_nk[n, k])
            self.model.addConstr(expr1 == 1, name="Constr1")

        for n1 in range(1, self.data.N + 1):
            for k in range(self.data.K):
                expr2_1 = grb.LinExpr(0)
                expr2_2 = grb.LinExpr(0)
                for n in range(1, self.data.N + 2):
                    expr2_1.addTerms(1, self.beta_nn1k[n1, n, k])
                for n in range(0, self.data.N + 1):
                    expr2_2.addTerms(1, self.beta_nn1k[n, n1, k])
                self.model.addConstr(expr2_1 == expr2_2, name="Constr2_1")
                self.model.addConstr(expr2_1 == self.alpha_nk[n1, k], name="Constr2_2")

        for k in range(self.data.K):
            expr3_1 = grb.LinExpr(0)
            expr3_2 = grb.LinExpr(0)
            for n in range(0, self.data.N + 1):
                expr3_1.addTerms(1, self.beta_nn1k[n, self.data.N + 1, k])
            for n in range(1, self.data.N + 2):
                expr3_2.addTerms(1, self.beta_nn1k[0, n, k])
            self.model.addConstr(expr3_1 == expr3_1, name="Constr3_1")
            self.model.addConstr(expr3_1 == 1, name="Constr3_2")

        for n in range(1, self.data.N + 1):
            for n1 in range(0, self.data.N):
                for k in range(self.data.K):
                    self.model.addConstr(self.gamma_nk[n, k] >= self.gamma_nk[n1, k] + self.data.t_nn1[n][n1] -
                                         self.data.M * (1 - self.beta_nn1k[n1, n, k]), name="Constr4")

    def add_terms(self):

        for k in range(self.data.K):
            expr5 = grb.LinExpr(0)
            for n in range(1, self.data.N):
                expr5.addTerms(self.data.q_n[n], self.alpha_nk[n, k])
        # OBJ
        obj = grb.LinExpr(0)
        for k in range(self.data.K):
            for n in range(0, self.data.N):
                for n1 in range(1, self.data.N + 1):
                    obj.addTerms(self.data.t_nn1[n][n1], self.beta_nn1k[n, n1, k])
        self.model.setObjective(obj, grb.GRB.MAXIMIZE)

    def solve(self):

        self.add_variables()
        self.add_constraints()
        self.add_terms()

        self.model.optimize()

        if self.isprint:
            self.print_info()

    def print_info(self):

        if self.model.status == grb.GRB.Status.INFEASIBLE:
            print("CVRP IS INFEASIBLE")
            self.model.computeIIS()
            self.model.write("CVRP + {0}.ilp".format(time.time()))
            return self.model
        elif self.model.status == grb.GRB.TIME_LIMIT:
            print("RUN OUT OF TIME LIMIT")
            return self.model
        elif self.model.status == grb.GRB.OPTIMAL:
            print("solveByGurobi_obj = {0}".format(self.model.ObjVal))
        
            def printVars():
                """
                print the optimal solution
                :return:
                """
                for n in range(1, self.data.N + 1):
                    for k in range(self.data.K):
                        if self.gamma_nk[n, k].x > 0:
                            print("gamma_nk[{0}][{1}] = {2}".format(n, k, self.gamma_nk[n, k].x))
        
                for n in range(1, self.data.N + 1):
                    for k in range(self.data.K):
                        if self.alpha_nk[n, k].x > 0.9:
                            print("alpha_nk[{0}][{1}] = {2}".format(n, k, self.alpha_nk[n, k].x))
        
                for n in range(self.data.N + 2):
                    for n1 in range(self.data.N + 2):
                        for k in range(self.data.K):
                            if self.beta_nn1k[n, n1, k].x > 0.9:
                                print("beta_nn1k[{0}][{1}][{2}] = {3}".format(n, n1, k, self.beta_nn1k[n, n1, k].x))
            printVars()
    
