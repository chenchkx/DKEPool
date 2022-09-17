

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.autograd import Function
import numpy as np


# Fast Matrix Power Normalized SPD Matrix Function
class FastMPNSPDMatrixFunction(Function):
    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False, device=x.device).type(dtype)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, iterN, 1, 1).type(dtype)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            YZY = 0.5 * Y[:, iterN - 2, :, :].bmm(I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]))
        y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, YZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :])) -
                          Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(Y[:, iterN - 2, :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) -
                               Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) -
                               ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) -
                               Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) -
                               dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i] \
                                    - grad_aux[i] / (normA[i] * normA[i])) \
                                   * torch.ones(dim, device=x.device).diag().type(dtype)
        return grad_input, None

class FastMPNSPDMatrix(nn.Module):
    def __init__(self, iterN = 5):
        super(FastMPNSPDMatrix, self).__init__()
        self.iterN = iterN

    def forward(self, input):
        output = FastMPNSPDMatrixFunction.apply(input, self.iterN)
        return output
