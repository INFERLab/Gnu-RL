import torch
from . import util

# @profile
def pnqp(H, q, lower, upper, x_init=None, n_iter=20):
    GAMMA = 0.1
    n_batch, n, _ = H.size()
    pnqp_I = 1e-11*torch.eye(n).type_as(H).expand_as(H)


    def obj(x):
        return 0.5*util.bquad(x, H) + util.bdot(q, x)

    if x_init is None:
        if n == 1:
            x_init = -(1./H.squeeze(2))*q
        else:
            H_lu = torch.linalg.lu_factor(H)
            LU, pivots = H_lu
            x_init = -torch.linalg.lu_solve(LU, pivots, q.unsqueeze(-1)).squeeze(-1)
    else:
        x_init = x_init.clone() # Don't over-write the original x_init.

    x = util.eclamp(x_init, lower, upper)

    # Active examples in the batch.
    J = torch.ones(n_batch, dtype=torch.bool, device=x.device)

    for i in range(n_iter):
        g = util.bmv(H, x) + q
        Ic = ((x == lower) & (g > 0)) | ((x == upper) & (g < 0))
        If = ~Ic

        Hff_I = util.bger(If.float(), If.float()).bool()
        not_Hff_I = ~Hff_I
        Hfc_I = util.bger(If.float(), Ic.float()).bool()

        g_ = g.clone()
        g_[Ic] = 0.
        H_ = H.clone()
        H_[not_Hff_I] = 0.0
        H_ += pnqp_I

        if n == 1:
            dx = -(1./H_.squeeze(2))*g_
        else:
            H_lu_ = torch.linalg.lu_factor(H_)
            LU_, pivots_ = H_lu_
            dx = -torch.linalg.lu_solve(LU_, pivots_, g_.unsqueeze(-1)).squeeze(-1)

        J = torch.norm(dx, 2, 1) >= 1e-4
        m = J.sum().item() # Number of active examples in the batch.
        if m == 0:
            return x, H_ if n == 1 else H_lu_, If, i

        alpha = torch.ones(n_batch).type_as(x)
        decay = 0.1
        max_armijo = GAMMA
        count = 0
        while max_armijo <= GAMMA and count < 10:
            # Crude way of making sure too much time isn't being spent
            # doing the line search.
            # assert count < 10

            maybe_x = util.eclamp(x+torch.diag(alpha).mm(dx), lower, upper)
            armijos = (GAMMA+1e-6)*torch.ones(n_batch).type_as(x)
            armijos[J] = (obj(x)-obj(maybe_x))[J]/util.bdot(g, x-maybe_x)[J]
            I = armijos <= GAMMA
            alpha[I] *= decay
            max_armijo = torch.max(armijos)
            count += 1

        x = maybe_x

    # TODO: Maybe change this to a warning.
    print("[WARNING] pnqp warning: Did not converge")
    return x, H_ if n == 1 else H_lu_, If, i
