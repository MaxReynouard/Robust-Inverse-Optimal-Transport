import numpy as np
import scipy.optimize
import functools
from progressbar import * 

# KL divergence
def KL(a, b) :
    return (a * np.log(a / b)).sum()

def sinkhorn(C, mu, nu, eps = 1,max_iter = 100, criterion = 1e-8) :
    '''
    Sinkhorn distance, no overflow management (so eps not too small),
    inputs :
        C Cost matrix, (n, m)
        mu, nu, the distributions,(n,1), (m,1)
        eps regularization parameter
        maximum number of iteration
        and a convergence criterion

    outputs :
        Pi matrix (n,m), the regularized transport Plan,
        u, v the sinkhorn support vectors
        the total cost : (C*Pi).sum() + regularization
        a text to print some informations on the run
    '''
    # start with checking if solvable transport problem
    assert np.abs(mu.sum()-nu.sum()) < criterion, "the distributions don't have the same sum "
    n,m = C.shape
    # initialise dual problem vectors
    u, v = np.ones((n,1))/n, np.ones((m,1))/m
    K = np.exp(-C / eps)
    i, converged = 0, False

    while i < max_iter and not converged :
        i+=1
        u_n = mu / (K @ v)
        v_n = nu / (K.T @ u_n)
        converged = np.linalg.norm(u_n-u) + np.linalg.norm(v_n-v) < criterion
        u, v = u_n, v_n

    # once done we can compute the transport plan
    Pi = u * K * v.T
    # and the regularized cost
    regul_cost = (C * Pi).sum() + eps * (Pi * np.log(Pi)).sum() - eps

    text = f'''the final transport plan is :
    {Pi}

    the regularized cost :
    {regul_cost}

    the dual support vectors :
    {u.T}
    {v.T}

    did we converge ? : {'yes' if converged else 'No'}'''
    if converged : text += f'\n number of iterations : {i}'
    return Pi, u, v, regul_cost, text

def sinkhorn_stab(C, mu, nu, eps = 1,max_iter = 100, criterion = 1e-8) :
    '''
    Sinkhorn distance, log stabilized,
    inputs :
        C Cost matrix, (n, m)
        mu, nu, the distributions,(n,1), (m,1)
        eps regularization parameter
        maximum number of iteration
        and a convergence criterion

    outputs :
        Pi matrix (n,m), the regularized transport Plan,
        f, g the recovery, and delivery costs (n,1) (m,1)
        the total cost : (C*Pi).sum() + regularization
        a text to print some informations on the run
    '''
    # to ease log stabilized computations
    def max_eps(z,epsilon) :
        '''
        epsilon boosted softmax on z columns,
        input :
            z of size (n,m)
        output :
            vector of size m, shape = (m,1)
        '''
        return epsilon * np.log(np.exp(z / epsilon).sum(0))[:, None]

    # start with checking if solvable transport problem
    assert np.abs(mu.sum()-nu.sum()) < criterion, "the distributions don't have the same sum "
    n,m = C.shape
    # initialise dual problem vectors (in log space)
    f = eps * np.log(np.ones((n,1))/n)
    g = eps * np.log(np.ones((m,1))/m)
    i, converged = 0, False

    # compute once since used in loop
    mu_ln = np.log(mu)
    nu_ln = np.log(nu)

    while i < max_iter and not converged :
        i+=1
        f_n = eps * mu_ln - max_eps(g - C.T, eps)
        g_n = eps * nu_ln - max_eps(f_n - C, eps)

        if i % 10 == 0 :
            u_diff = np.linalg.norm(np.exp(f / eps) -  np.exp(f_n / eps))
            v_diff = np.linalg.norm(np.exp(g / eps) -  np.exp(g_n / eps))
            converged = u_diff + v_diff < criterion
        f, g = f_n, g_n

    Pi = np.exp((f + g.T - C) / eps)

    regul_cost = (C * Pi).sum() + eps * (Pi * np.log(Pi)).sum() - eps

    text = f'''the final transport plan is :
    {Pi}

    the regularized cost :
    {regul_cost}

    the dual vectors :
    {f.T}
    {g.T}

    did we converge ? : {'yes' if converged else 'No'}'''
    if converged : text += f'\n number of iterations : {i}'
    return Pi, f, g, regul_cost, text

# polynomial kernel to compute cost matrix C from A
def poly_kernel (U, V, gamma, c0, d, A):
    return gamma * (U @ A @ V.T + c0) ** d

def RIOT(Pi_hat, C_u, C_v, C_of_A, Delta_of_A,
         lambda0, lambda_u,lambda_v, delta , s,
         outer_iter, inner_iter, UV_features = None, A = None,
          Pi_real = None) :
    r""" Robust Inverse Optimal Transport

    inputs :
        observed matching matrix Pi_hat,
        user-user cost matric C_u,
        item-item cost matric C_v,
        kernel representation (depending on users and items positions in their features space) C_of_A,
        its derivative dC_of_A,
        regularization parameters : lambda0, lambda_u, lambda_v, delta,
        learning rate s,
        number of inner and outer iterations
        initialised A matrix OR number of features for UV (an int or a pair of ints)
        if both are given, A is prioritised
        can also give the real transport plan : Pi_real, to compute the distance to Pi as loops run

    outputs:
        Pi our reconstructed transport Plan
        C the reconstructed cost matrix
        KLs the KL divergence (Pi,Pi_hat)
        PiDists the distance from Pi_real to Pi as loops run
    """
    # to ease inner loop computations
    def find_root (u_hat,Z,eta,M):
        def p(theta) :
            return ((u_hat * (Z @ eta)) / ((M - theta * Z) @ eta)).sum()-1
        mini_pole = np.min(M @ eta / Z @ eta)
        return scipy.optimize.root(p, mini_pole - 10)

    # after inner loop, to check if computation errors
    # didn't deviate us too much from integrity
    def check_constraints(M, eta, ksi, mu_hat, nu_hat, theta, Z, tol = 1e-8) :
        '''
        those values needs to be close to 0
        for the loop the be able to keep running
        '''
        c1 = np.linalg.norm(M @ eta - mu_hat / ksi - theta * Z @ eta)
        c2 = np.linalg.norm(M.T @ ksi - nu_hat / eta - theta * Z.T @ ksi)
        c3 = ksi.T @ Z @ eta - 1
        c = c1 > tol or c2 > tol or c3 > tol
        if c : print(f'exited outer loop because of of the constraint is more than {tol} : {c1, c2, c3}')
        return c

    if A is None :
        if UV_features is None :
            print('we need at list UV_features or A in RIOT call')
            return
        else :
            u_feat,v_feat = [UV_features]*2 if type(UV_features)==int else UV_features
            A = np.random.rand(u_feat,v_feat)

    C = C_of_A(A)
    mu_hat, nu_hat = Pi_hat.sum(1)[:,None], Pi_hat.sum(0)[:,None]
    z = sinkhorn_stab(C_u, mu_hat, mu_hat, 1 / lambda_u)[2]
    w = sinkhorn_stab(C_v, nu_hat, nu_hat, 1 / lambda_v)[2]
    Pi, ksi, eta = sinkhorn_stab(C,mu_hat,nu_hat)[:3]
    ksi = np.exp(ksi * lambda0)
    eta = np.exp(eta * lambda0)

    if Pi_real is not None :
        PiDists = [np.linalg.norm(Pi-Pi_real)]
    KLs = [KL(Pi, Pi_hat)]
    best_KL = np.inf
    print("Performing RIOT :")
    pbar = ProgressBar(maxval=outer_iter)
    pbar.start()
    for _ in range(outer_iter) :
        Z = np.exp(- C * lambda0)
        M = delta * (z + w.T) * Z

        for __ in range(inner_iter):
            recenter = (eta.mean() / ksi.mean()) ** 0.5
            ksi_n = ksi * recenter
            eta_n = eta / recenter
            theta_1 = find_root(mu_hat,Z,eta_n,M).x[0]
            ksi_n = mu_hat / ((M - theta_1 * Z) @ eta_n)
            theta_2 = find_root(nu_hat,Z.T,ksi_n,M.T).x[0]
            eta_n = nu_hat / ((M - theta_2 * Z).T @ ksi_n)
            if np.linalg.norm(ksi_n-ksi) + np.linalg.norm(eta_n-eta)< 1e-9 : break
            eta, ksi = eta_n, ksi_n

        constraints = check_constraints(M, eta, ksi, mu_hat, nu_hat, theta_2, Z)
        if constraints :
            print('failed in outer iteration', _)
            print('M :\n',M,'\n')
            print('eta :\n',eta,'\n')
            print('ksi :\n',ksi,'\n')
            print('mu_hat :\n',mu_hat,'\n')
            print('nu_hat :\n',nu_hat,'\n')
            print('theta_2 :\n',theta_2,'\n')
            print('Z :\n',Z,'\n')
            break

        Pi = ksi * Z @ np.diagflat(eta)

        grad_C = (Pi_hat + (theta_2 - delta * (z + w.T)) * Pi) * lambda0
        Delta_A = Delta_of_A(A,grad_C)
        A -= s * Delta_A
        C = C_of_A(A)

        z = sinkhorn_stab(C_u, Pi.sum(1)[:, None], mu_hat, 1 / lambda_u)[2]
        w = sinkhorn_stab(C_v, Pi.sum(0)[:, None], nu_hat, 1 / lambda_v)[2]
        if Pi_real is not None :
            PiDists.append(np.linalg.norm(Pi-Pi_real))
        KLs.append(KL(Pi, Pi_hat))
        pbar.update(_)
    pbar.finish()
    if Pi_real is not None :
        return Pi, C, A, KLs, PiDists
    return Pi, C, A, KLs
