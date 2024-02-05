import math
import numpy as np
import torch


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    x = torch.tensor(x, device='cuda:0').cpu().numpy() # 추가
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)

        # # dx = (x - self.x_prev) / t_e
        # dx = (torch.tensor(x, device='cuda:0') - torch.tensor(self.x_prev, device='cuda:0')).cpu().numpy().astype(np.float32) / t_e # 수정
        # dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # # The filtered signal.
        # cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        # a = smoothing_factor(t_e, cutoff)
        # x_hat = exponential_smoothing(a, x, self.x_prev)

         # x와 self.x_prev를 PyTorch 텐서로 변환합니다.
        x_torch = torch.tensor(x, device='cuda:0').cpu().numpy()
        x_prev_torch = torch.tensor(self.x_prev, device='cuda:0').cpu().numpy()

        # dx를 계산합니다.
        dx = (x_torch - x_prev_torch) / t_e  # 여기에서 PyTorch 텐서로 변환할 필요가 없습니다.

        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # 필터링된 신호.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)

        # x를 PyTorch 텐서로 변환합니다.
        x_torch = torch.tensor(x, device='cuda:0').cpu().numpy()

        x_hat = exponential_smoothing(a, x_torch, x_prev_torch)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
