import numpy as np
import math
from scipy.optimize import linprog
import time
import warnings
warnings.filterwarnings("ignore")

def is_int(X, thr=1.0E-12):
    """
    X: np.array(n,)
    If X is an int vector return (True, -1) else return (False, first index of fraction)
    """
    for i, x in enumerate(X):
        if (x-math.floor(x)) > thr and (math.ceil(x)-x) > thr:
            return (False, i)
    return (True, -1)

def make_int(X, thr=1.0E-12):
    """
    X: np.array(n,)
    If X is an int vector then make it strictly int
    """
    assert is_int(X)[0]
    for i, x in enumerate(X):
        if x - math.floor(x) < thr:
            X[i] = math.floor(x)
        elif math.ceil(x) - x < thr:
            X[i] = math.ceil(x)
    return X

class Tree(object):
    def __init__(self, dic):
        self.val = dic  # dic = {'A': A, 'b': b}, used to store regions A.dot(x) <= b
        self.left = None
        self.right = None

    def tree_print(self, n_indent=0):
        prefix = '  ' * n_indent
        print(prefix + 'root:', self.val)
        if self.left is not None:
            print(prefix + 'left:')
            self.left.tree_print(n_indent+1)
        if self.right is not None:
            print(prefix + 'right:')
            self.right.tree_print(n_indent+1)
        print(prefix + 'end')

    def all_leaves(self, A=None, b=None):
        """
        Enumerate all leaves. Each leaf is accumulated from its parents:
        A = union of all A's in nodes from root to leaf; similar to b
        """
        result = []
        if A is None and b is None:
            A, b = self.val['A'], self.val['b']
        else:
            A = np.vstack([A, self.val['A']])
            b = np.append(b, self.val['b'])
        if self.left is None and self.right is None:
            result.append({'A': A, 'b': b})
        else:
            if self.left is not None:
                result += self.left.all_leaves(A, b)
            if self.right is not None:
                result += self.right.all_leaves(A, b)
        return result

class MILP(object):
    def __init__(self, parameters):
        """
        parameters is a dict including c, A, b, Aeq, beq, k, bounds
        min     c.dot(x) 
        s.t.    Aeq.dot(x) = beq
                A.dot(x) < b
                x in bounds
                first k elements in x are int
        """
        self.parameters = parameters
        dic = {'A': parameters['A'], 'b': parameters['b']}
        self.T = Tree(dic)  # store splitting regions from branch-and-bound
        self.bestVal, self.bestX, self.feas = None, None, False
        self.all_sol = set([])  # set of all feasible int solutions (restricted to the first k elements)
        self.timetable = [[], []]  # [list of seconds, list of #linear regions found]

    def solveMILP(self):
        c = self.parameters['c']
        A = self.parameters['A']
        b = self.parameters['b']
        Aeq = self.parameters['Aeq']
        beq = self.parameters['beq']
        k = self.parameters['k']
        bounds = self.parameters['bounds']
        self.bestVal, self.bestX, self.feas = self._solveMILP(c, A, b, Aeq, beq, k, bounds=bounds)

    def print_result(self, do_print_tree=False):
        print('best value:', self.bestVal)
        try:
            print('best x:', self.bestX[:self.parameters['k']])
        except:
            print('best x:', self.bestX)
        print('feasibility:', self.feas)
        if do_print_tree:
            print('Branching tree:')
            self.T.tree_print()
        print('number of solutions:', len(self.all_sol))
    
    def _solveMILP(self, c, A, b, Aeq, beq, k, bounds=None):
        """
        Mixed integer programing solving 
        min     c.dot(x) 
        s.t.    Aeq.dot(x) = beq
                A.dot(x) < b
                x in bounds
                first k elements in x are int
        return best value, best x, and splitting tree for ineq constraints over int vars
        """
        # dimension check
        d = c.shape[0]
        assert k <= d
        N_ineq = A.shape[0]
        N_eq = Aeq.shape[0]
        assert c.shape == (d, )
        assert A.shape == (N_ineq, d)
        assert b.shape == (N_ineq, )
        assert Aeq.shape == (N_eq, d)
        assert beq.shape == (N_eq, )
        assert len(bounds) == d

        # solve the relaxed linear program problem
        try:
            res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='interior-point')
        except:
            res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='interior-point', options={'lstsq': True})
        bestVal = float('inf')
        bestX = res.x

        if not(type(res.x) is float or res.status != 0): 
            bestVal = c.dot(bestX)
        else:
            return (bestVal, bestX, False)

        if k == 0 or is_int(bestX[:k])[0]:
            # if the solution is feasible then return it
            return (bestVal, bestX, True)
        else:
            # find a fractional index
            ind = is_int(bestX[:k])[1]

            # split into 2 branches
            newCon1 = np.zeros(d)
            newCon1[ind] = -1
            newA1 = np.vstack([A, newCon1])
            newB1 = np.append(b, [-math.ceil(bestX[ind])])
            dic1 = {'A': newCon1, 'b': [-math.ceil(bestX[ind])]}
            
            newCon2 = np.zeros(d)
            newCon2[ind] = 1
            newA2 = np.vstack([A, newCon2])
            newB2 = np.append(b, [math.floor(bestX[ind])])
            dic2 = {'A': newCon2, 'b': [math.floor(bestX[ind])]}
            
            # solve the two sub problems
            r1 = self._solveMILP(c, newA1, newB1, Aeq, beq, k, bounds=bounds)
            r2 = self._solveMILP(c, newA2, newB2, Aeq, beq, k, bounds=bounds)

            # if not feasible then do not record this branch
            if r1[2]:
                self.T.left = Tree(dic1)
            if r2[2]:
                self.T.right = Tree(dic2)

            # return the better one
            if r1[0] < r2[0]:
                return r1
            else:
                return r2

    def onetree_phase2(self):
        """
        Find all solutions to the mixed integer programing problem
        (One-tree algorithm phase-2)
        min     c.dot(x) 
        s.t.    Aeq.dot(x) = beq
                A.dot(x) < b
                x in bounds
                first k elements in x are int
        return the set of all solutions
        """

        def find_all_sol_each_node(c, A, b, Aeq, beq, k, bounds=None):
            d = c.shape[0]

            # solve MILP and obtain bestX
            all_sol = set([])
            try:
                try:
                    res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='interior-point')
                except:
                    res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='interior-point', options={'lstsq': True})
            except:
                return all_sol
            bestX = res.x

            if type(res.x) is float or res.status != 0:
                return all_sol
            
            # try to split, find ind and split_point
            can_do_split = False
            if is_int(bestX[:k])[0]:
                all_sol.add(tuple(make_int(bestX[:k])))

                # find ind such that x[ind] is not tight
                try:
                    res_dual = linprog(np.zeros(d), A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='interior-point')
                except:
                    res_dual = linprog(np.zeros(d), A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='interior-point', options={'lstsq': True})
                if res_dual.status != 0:
                    return all_sol
                bestX_dual = res_dual.x

                # find index such that bestX[ind] != bestX_dual[ind]
                for i in range(k):
                    if abs(bestX[i] - bestX_dual[i]) > 1e-3:
                        can_do_split = True
                        ind = i
                        split_point = (bestX[ind] + bestX_dual[ind]) / 2
                        break
            else:
                can_do_split = True
                ind = is_int(bestX[:k])[1]
                split_point = bestX[ind]

            # split using ind and split_point: x[ind] < split_point and x[ind] > split_point
            if can_do_split:
                newCon1 = np.zeros(d)
                newCon1[ind] = -1
                newA1 = np.vstack([A, newCon1])
                newB1 = np.append(b, [-math.ceil(split_point)])
                
                newCon2 = np.zeros(d)
                newCon2[ind] = 1
                newA2 = np.vstack([A, newCon2])
                newB2 = np.append(b, [math.floor(split_point)])
            
                all_sol |= find_all_sol_each_node(c, newA1, newB1, Aeq, beq, k, bounds=bounds)
                all_sol |= find_all_sol_each_node(c, newA2, newB2, Aeq, beq, k, bounds=bounds)
                self.all_sol |= all_sol

            # update timetable
            if not self.timetable[1] or len(self.all_sol) > self.timetable[1][-1]:
                self.timetable[0].append(np.around(time.time() - self.time0, decimals=2))
                self.timetable[1].append(len(self.all_sol))
            return all_sol
        

        c = self.parameters['c']
        Aeq = self.parameters['Aeq']
        beq = self.parameters['beq']
        k = self.parameters['k']
        bounds = self.parameters['bounds']
        regions = self.T.all_leaves()
        
        self.time0 = time.time()

        assert k > 0
        for dic in regions:
            A, b = dic['A'], dic['b']
            find_all_sol_each_node(c, A, b, Aeq, beq, k, bounds=bounds)

def generate_FC_parameters(d, hlist, x_range=1.0):
    """
    Generate parameters for a FC network
    d = input dimension
    hlist = size of network: [m_1, m_2,..., m_L]
    search region = [-x_range, x_range]
    """
    W = []
    beq = []
    L = len(hlist)  # number of hidden layers, including output layer
    k = sum(hlist)  # number of neurons

    # randomly generate weight matrices
    for i in range(L):
        if i == 0:
            W.append(np.random.rand(hlist[i], d) - 0.5)
        else:
            W.append(np.random.rand(hlist[i], hlist[i-1]) - 0.5)
        beq = np.append(beq, np.random.rand(hlist[i]) - 0.5)

    # large enough M
    # TODO: can be tightened
    M = x_range * d * np.product(hlist)

    c = np.append(np.zeros(3*k+d), [1])

    Aeq = np.zeros([k, 3*k+d+1])
    for i in range(k):
        Aeq[i][k+i] = 1
        Aeq[i][2*k+i] = -1
    for j in range(hlist[0]):
        Aeq[j][(3*k):(3*k+d)] = -W[0][j]
    for i in range(1, L):
        acc = sum(hlist[:i-1])
        for j in range(hlist[i]):
            row = j + acc + hlist[i-1]
            Aeq[row][(k+acc):(k+acc+hlist[i-1])] = -W[i][j]

    A = np.zeros([3*k, 3*k+d+1])
    for i in range(k):
        A[i][i] = -M
        A[i][i+k] = 1
        A[i+k][i] = M
        A[i+k][i+2*k] = 1
        A[i+2*k][i] = M
        A[i+2*k][i+k] = -1
        A[i+2*k][3*k+d] = 1
    b = np.append(np.zeros(k), M*np.ones(2*k))
    bounds = tuple([(0,1,)]*k + [(0,M,)]*(2*k) + [(-x_range,x_range,)]*d + [(0,M,)])

    parameters = {  'c': c,
                    'A': A,
                    'b': b,
                    'Aeq': Aeq,
                    'beq': beq,
                    'k': k,
                    'bounds': bounds}
    return parameters

parameters = generate_FC_parameters(d=3, hlist=[2,20,10], x_range=5.0)
problem = MILP(parameters)
time0 = time.time()

problem.solveMILP()
time1 = time.time()

problem.onetree_phase2()
time2 = time.time()

print('#linear regions:', len(problem.all_sol))
print('solve MILP:', time1-time0, 'seconds')
print('onetree phase 2:', time2-time1, 'seconds')

