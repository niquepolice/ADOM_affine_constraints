import numpy as np
import utils
import scipy.linalg
import cvxpy as cp

class Model:
    def __init__(self, nodes, dim, cons_rows=2, mu=1, L=1000, lmaxATA=2, graph_model="ring", edge_prob=None):
        self.nodes, self.dim = nodes, dim
        self.mu, self.L = mu, L
        
        self.graph_model = graph_model
        self.edge_prob = edge_prob
        
        if self.graph_model == "ring":
            k = np.arange(0, self.nodes // 2 + 1)
            spectrum = 2 - 2 * np.cos(2 * np.pi * k / self.nodes)
            self.chi = spectrum.max() / spectrum[1:].min() * 1.01 # for numerical robustness
            # print("graph chi:", self.chi)
        elif self.graph_model == "erdos-renyi":
            self.chi = 10 # some random constant
        self.get_bW() # check chi
            
        self.C = []  # np.random.random((nodes, dim, dim))
        for i in range(self.nodes):
            Ci = np.random.random((dim-1, dim)) # mu = 0
            Ci = Ci.T @ Ci
            Ci *= (self.L - self.mu) / utils.lambda_max(Ci)
            Ci += self.mu * np.identity(dim) 
            self.C.append(Ci)
        # print("mu=", min([utils.lambda_min(Ci) for Ci in self.C]))
        self.bC = scipy.linalg.block_diag(*self.C)

        # multiply by mu to preserve scaling and numeric stability
        self.d = [np.random.random(self.dim) * self.mu for _ in range(self.nodes)]
        self.bd = np.hstack(self.d)
        
        self.Csum = sum(self.C)
        self.dsum = sum(self.d)
        
        self.cons_rows = cons_rows
        
        lminpATA = 1
        A = utils.get_matrix(cons_rows, dim, np.linspace(lminpATA, lmaxATA, cons_rows))
        self.A = A / lmaxATA ** 0.5 
        self.chitA = lmaxATA / lminpATA
        # print("chitA", self.chitA)
            
        s2maxA = 1
        s2minpA = 1 / self.chitA

        # uncomment this and last lines to make Ax = b <=> 0 = 0 
        # s2maxA = 0
        # s2minpA = 0 
        # self.A  = np.zeros(self.A.shape)
        # self.chitA = lminpATA = lmaxATA = 0

        self.muH = (1 + s2minpA) / self.L
        self.LH = (1 + s2maxA) / self.mu
        # print(f"muH {self.muH}, LH {self.LH}")
        
        In = np.identity(self.nodes)
        self.bA = np.kron(In, self.A)
        self.b = np.random.random(self.cons_rows) 
        self.bb = np.hstack([self.b] * self.nodes)
        
        Id = np.identity(self.dim)
        ones = np.ones((self.nodes, self.nodes))
        self.P = np.kron(In - (ones / self.nodes), Id) # projector on L^\bot
        
        self.f_star, self.x_star = self._get_solution()
        self.f_star_cons, self.x_star_cons = self._get_cons_solution()
        # if A = 0, b = 0
        # self.f_star_cons, self.x_star_cons = self._get_solution()
    
    def split(self, x):
        return x.reshape((self.nodes, self.dim))
    
    def get_bW(self):
        if self.graph_model == "ring":
            W = utils.get_ring_W(self.nodes)  # graph Laplacian. 
        elif self.graph_model == "erdos-renyi":
            W = utils.get_ER_W(self.nodes, self.edge_prob)
            
        perm = np.random.permutation(self.nodes)
        W = W[perm].T[perm].T
        W /= utils.lambda_max(W)
        if not self.chi >= 1 / utils.lambda_min_plus(W):  # check that chi is correctly choosen
            print("eigvals", sorted(np.linalg.eigvals(W)))
            print(f"chi: {self.chi}, actual lmax per lminp: {1 / utils.lambda_min_plus(W)}")
            assert False
        
        Id = np.identity(self.dim)
        self.W = W
        return np.kron(W, Id)
        

    def F(self, bx):
        return (1 / 2 * bx.T @ self.bC @ bx + self.bd @ bx)
    
    def f(self, x):
        return (1 / 2 * x.T @ self.Csum @ x + self.dsum @ x)

    def grad_F(self, bx):
        return (self.bC @ bx + self.bd)

    def _get_solution(self):
        x_star = np.linalg.solve(self.Csum, -self.dsum)
        return self.f(x_star), x_star
    
    def _get_cons_solution(self):
        x = cp.Variable(self.dim)
        f = cp.quad_form(x, self.Csum) / 2 + self.dsum @ x
        cons = [self.A @ x == self.b]
        obj = cp.Minimize(f)
        prob = cp.Problem(obj, cons).solve(verbose=False, eps_abs=1e-10)
        x_star = x.value
        return self.f(x_star), x_star

    def Fstar_grad(self, bu):
        # -(z.T @ t - F(x)) -> min_x
        return np.linalg.solve(
            self.bC, self.bd - bu
        )


def ADOM_affine(iters: int, model: Model, inner_steps=1):
    lminp = 1 / model.chi
    
    alpha = 0.5 * model.muH 
    eta = 2. * lminp / np.sqrt(model.LH * model.muH) / 7.
    theta = 1 / model.LH
    sigma = 1.
    tau = lminp * np.sqrt(model.muH / model.LH) / 7.
    
    p_dim = model.nodes * model.cons_rows
    s_dim = model.nodes * model.dim
    z_dim = p_dim + s_dim
    z = np.zeros(z_dim)
    z_f = np.zeros(z_dim)
    m = np.zeros(z_dim)
    g = np.zeros(s_dim)
    
    f_err, cons_err, affine_cons_err, dist = np.zeros(iters), np.zeros(iters), np.zeros(iters), np.zeros(iters)
    
    bW = np.identity(z_dim) # initilize without laplacian
    B = np.vstack((model.bA, np.identity(s_dim)))
    q = np.hstack((model.bb, np.zeros(s_dim)))
    
    for i in range(iters):
        bW[p_dim:, p_dim:] = model.get_bW()
        z_g = tau * z + (1. - tau) * z_f     
    
        # inexact dual oracle via gradient step. g \aprox \nabla F*(B^T z_g)
        BTzg = B.T @ z_g
        for _ in range(inner_steps):
            g -= (model.grad_F(g) - BTzg) / model.L
        dH = B @ g - q
    
        delta = sigma * bW @ (m - eta * dH)
        
        m -= eta * dH + delta
        z += eta * alpha * (z_g - z) + delta
        z_f = z_g - theta * bW @ dH 
        
        f_err[i] = model.F(g) - model.f_star_cons
        cons_err[i] = np.linalg.norm(model.P @ g)
        affine_cons_err[i] = np.linalg.norm(model.bA @ g - model.bb)

        dist[i] = np.sum((g.reshape(model.nodes, model.dim) - model.x_star_cons) ** 2, axis=1).max() ** 0.5

    return g, z_f, f_err, cons_err, affine_cons_err, dist
   
