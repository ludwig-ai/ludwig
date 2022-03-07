from typing import Optional

from einops import rearrange, repeat
import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F

from ludwig.modules.fully_connected_modules import FCLayer
from ludwig.utils.torch_utils import get_activation, LudwigModule


contract = oe.contract
contract_expression = oe.contract_expression


class S4(LudwigModule):
    def __init__(
        self,
        model_size: int,
        hidden_size: int,
        output_size: int = None,
        sequence_size: Optional[int] = 1,
        num_channels: int = 1,
        bidirectional: bool = False,
        activation: str = "gelu",
        dropout: float = 0,
        fc_use_bias=True,
        fc_weights_initializer="xavier_uniform",
        fc_bias_initializer="zeros",
        fc_norm=None,
        fc_norm_params=None,
    ):
        self.h = model_size
        self.n = hidden_size
        self.l_max = sequence_size
        self.bidirectional = bidirectional
        self.num_channels = num_channels

        self.D = nn.Parameter(torch.randn(self.num_channels, self.h))

        if self.bidirectional:
            self.num_channels *= 2

        # SSM Kernel
        self.kernel = HippoSSKernel(self.h, N=self.n, L=self.l_max, channels=self.num_channels)

        # Pointwise
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_size = output_size if output_size is not None else self.h
        self.output_linear = FCLayer(
            self.h * self.channels,
            input_rank=3,
            output_size=self.output_size,
            use_bias=fc_use_bias,
            weights_initializer=fc_weights_initializer,
            bias_initializer=fc_bias_initializer,
            norm=fc_norm,
            norm_params=fc_norm_params,
            activation=None,
        )

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.l_max:
            return torch.Size([self.l_max, self.output_size])
        return self.output_size

    def forward(self, u):
        # if not self.transposed:
        #     u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        k = self.kernel(L=L)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=2 * L)  # (C H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y_f = contract("b h l, c h l -> b c h l", u_f, k_f)  # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract("b h l, c h -> b c h l", u, self.D)  # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Reshape to flatten channels
        y = rearrange(y, "... c h l -> ... (c h) l")
        y = self.dropout(self.activation(y))
        y = y.transpose(-1, -2)  # output shape (B L (C H))
        y = self.output_linear(y)  # output shape (B L H)
        return y


class HippoSSKernel(LudwigModule):
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.
    The SSKernel is expected to support the interface
    forward()
    default_state()
    setup_step()
    step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=1,
        measure="legs",
        rank=1,
        channels=1,  # 1-dim to C-dim map; can think of C as having separate "heads"
        dt_min=0.001,
        dt_max=0.1,
        trainable=None,  # Dictionary of options to train various HiPPO parameters
        lr=None,  # Hook to set LR of hippo parameters differently
        length_correction=True,  # Multiply by I-A|^L after initialization; can be turned off for initialization speed
        hurwitz=False,
        tie_state=False,  # Tie parameters of HiPPO ODE across the H features
        precision=1,  # 1 (single) or 2 (double) for the kernel
        resample=False,  # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
        verbose=False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
        self.rate = None if resample else 1.0
        self.channels = channels

        # Generate dt
        log_dt = torch.rand(self.H, dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        w, p, B, _ = self._nplr(measure, self.N, rank, dtype=dtype)
        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)
        self.kernel = SSKernelNPLR(
            L,
            w,
            p,
            B,
            C,
            log_dt,
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction,
            verbose=verbose,
        )

    def _nplr(measure, N, rank=1, dtype=torch.float):
        """
        Return w, p, q, V, B such that
        (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
        i.e. A = V[w - p q^*]V^*, B = V B
        """
        assert dtype == torch.float or torch.cfloat
        if measure == "random":
            dtype = torch.cfloat if dtype == torch.float else torch.cdouble
            # w = torch.randn(N//2, dtype=dtype)
            w = -torch.exp(torch.randn(N // 2)) + 1j * torch.randn(N // 2)
            P = torch.randn(rank, N // 2, dtype=dtype)
            B = torch.randn(N // 2, dtype=dtype)
            V = torch.eye(N, dtype=dtype)[..., : N // 2]  # Only used in testing
            return w, P, B, V

        A, B = transition(measure, N)
        A = torch.as_tensor(A, dtype=dtype)  # (N, N)
        B = torch.as_tensor(B, dtype=dtype)[:, 0]  # (N,)

        P = rank_correction(measure, N, rank=rank, dtype=dtype)
        AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)
        w, V = torch.linalg.eig(AP)  # (..., N) (..., N, N)
        # V w V^{-1} = A

        # Only keep one of the conjugate pairs
        w = w[..., 0::2].contiguous()
        V = V[..., 0::2].contiguous()

        V_inv = V.conj().transpose(-1, -2)

        B = contract("ij, j -> i", V_inv, B.to(V))  # V^* B
        P = contract("ij, ...j -> ...i", V_inv, P.to(V))  # V^* P

        return w, P, B, V

    def forward(self, L=None):
        k, _ = self.kernel(rate=self.rate, L=L)
        return k.float()


class SSKernelNPLR(LudwigModule):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)
    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.
    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix
    The forward pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)
    """

    @torch.no_grad()
    def _setup_C(self, double_length=False):
        """Construct C~ from C
        double_length: current C is for length L, convert it to length 2L
        """
        C = _r2c(self.C)
        self._setup_state()
        dA_L = power(self.L, self.dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length:
            prod = -prod  # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., : self.N]  # Take conjugate pairs again
        self.C.copy_(_c2r(C_))

        if double_length:
            self.L *= 2
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

    def _omega(self, L, dtype, device, cache=True):
        """Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
        This should be called everytime the internal length self.L changes"""
        omega = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=dtype, device=device)  # \omega_{2L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)
        if cache:
            self.register_buffer("omega", _c2r(omega))
            self.register_buffer("z", _c2r(z))
        return omega, z

    def __init__(
        self,
        L,
        w,
        P,
        B,
        C,
        log_dt,
        hurwitz=False,
        trainable=None,
        lr=None,
        tie_state=False,
        length_correction=True,
        verbose=False,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        w: (N)
        p: (r, N) low-rank correction to A
        q: (r, N)
        A represented by diag(w) - pq^*
        B: (N)
        dt: (H) timescale per feature
        C: (H, C, N) system is 1-D to c-D (channels)
        hurwitz: tie pq and ensure w has negative real part
        trainable: toggle which of the parameters is trainable
        lr: add hook to set lr of hippo parameters specially (everything besides C)
        tie_state: tie all state parameters across the H hidden features
        length_correction: multiply C by (I - dA^L) - can be turned off when L is large for slight speedup at initialization (only relevant when N large as well)
        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.hurwitz = hurwitz
        self.tie_state = tie_state
        self.verbose = verbose

        # Rank of low-rank correction
        self.rank = P.shape[-2]
        assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N)))  # (H, C, N)
        H = 1 if self.tie_state else self.H
        B = repeat(B, "n -> 1 h n", h=H)
        P = repeat(P, "r n -> r h n", h=H)
        w = repeat(w, "n -> h n", h=H)

        # Cache Fourier nodes every time we set up a desired length
        self.L = L
        if self.L is not None:
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

        # Register parameters
        # C is a regular parameter, not state
        # self.C = nn.Parameter(_c2r(C.conj().resolve_conj()))
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        train = False
        if trainable is None:
            trainable = {}
        if trainable == False:
            trainable = {}
        if trainable == True:
            trainable, train = {}, True
        self.register("log_dt", log_dt, trainable.get("dt", train), lr, 0.0)
        self.register("B", _c2r(B), trainable.get("B", train), lr, 0.0)
        self.register("P", _c2r(P), trainable.get("P", train), lr, 0.0)
        if self.hurwitz:
            log_w_real = torch.log(-w.real + 1e-3)  # Some of the HiPPO methods have real part 0
            w_imag = w.imag
            self.register("log_w_real", log_w_real, trainable.get("A", 0), lr, 0.0)
            self.register("w_imag", w_imag, trainable.get("A", train), lr, 0.0)
            self.Q = None
        else:
            self.register("w", _c2r(w), trainable.get("A", train), lr, 0.0)
            # self.register("Q", _c2r(P.clone().conj().resolve_conj()), trainable.get('P', train), lr, 0.0)
            Q = _resolve_conj(P.clone())
            self.register("Q", _c2r(Q), trainable.get("P", train), lr, 0.0)

        if length_correction:
            self._setup_C()

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.hurwitz:
            w_real = -torch.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = _r2c(self.w)  # (..., N)
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor
        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)

        # Increase the internal length if needed
        while rate * L > self.L:
            self.double_length()

        dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)
        w = self._w()

        if rate == 1.0:
            # Use cached FFT nodes
            omega, z = _r2c(self.omega), _r2c(self.z)  # (..., L)
        else:
            omega, z = self._omega(int(self.L / rate), dtype=w.dtype, device=w.device, cache=False)

        if self.tie_state:
            B = repeat(B, "... 1 n -> ... h n", h=self.H)
            P = repeat(P, "... 1 n -> ... h n", h=self.H)
            Q = repeat(Q, "... 1 n -> ... h n", h=self.H)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.size(-1) == self.N else state  # (B H N)
            sA = s * _conj(w) - contract("bhm, rhm, rhn -> bhn", s, _conj(Q), _conj(P))  # (B H N)
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., : self.N]

            B = torch.cat([s, B], dim=-3)  # (s+1, H, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, P], dim=-3)  # (s+1+r, H, N)
        C = torch.cat([C, Q], dim=-3)  # (c+r, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (s+1+r, c+r, H, N)
        # w = w[None, None, ...]  # (1, 1, H, N)
        # z = z[None, None, None, ...]  # (1, 1, 1, L)

        # Calculate resolvent at omega
        if has_cauchy_extension and z.dtype == torch.cfloat:
            r = cauchy_mult(v, z, w, symmetric=True)
        else:
            r = cauchy_conj(v, z, w)
        r = r * dt[None, None, :, None]  # (S+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f)  # (S+1, C, H, L)

        # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (S, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :]  # (C H L)
        return k_B, k_state

    @torch.no_grad()
    def double_length(self):
        if self.verbose:
            log.info(f"S4: Doubling length from L = {self.L} to {2*self.L}")
        self._setup_C(double_length=True)

    def _setup_linear(self):
        """Create parameters that allow fast linear stepping of state"""
        w = self._w()
        B = _r2c(self.B)  # (H N)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)

        # Prepare Linear stepping
        dt = torch.exp(self.log_dt)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        R = (
            torch.eye(self.rank, dtype=w.dtype, device=w.device)
            + 2 * contract("r h n, h n, s h n -> h r s", Q, D, P).real
        )  # (H r r)
        Q_D = rearrange(Q * D, "r h n -> h r n")
        R = torch.linalg.solve(R.to(Q_D), Q_D)  # (H r N)
        R = rearrange(R, "h r n -> r h n")

        self.step_params = {
            "D": D,  # (H N)
            "R": R,  # (r H N)
            "P": P,  # (r H N)
            "Q": Q,  # (r H N)
            "B": B,  # (1 H N)
            "E": 2.0 / dt.unsqueeze(-1) + w,  # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.
        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster
        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C)  # View used for dtype/device

        if u is None:  # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None:  # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if state.size(-1) == self.N:  # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract("r h n, r h m, ... h m -> ... h n", _conj(p), _conj(x), _conj(y))[
                ..., : self.N
            ]  # inner outer product
        else:
            assert state.size(-1) == 2 * self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: contract("r h n, r h m, ... h m -> ... h n", p, x, y)  # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state)  # (B H N)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """Construct dA and dB for discretized state equation"""

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C)  # Just returns a view that we use for finding dtype/device

        state = torch.eye(2 * self.N, dtype=C.dtype, device=C.device).unsqueeze(-2)  # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA  # (H N N)

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, "1 h n -> h n")  # (H N)

    def _step_state(self, u, state):
        """Must be called after self.default_state() is used to construct an initial state!"""
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state

    def setup_step(self, mode="dense"):
        """Set up dA, dB, dC discretized parameters for stepping"""
        self._setup_state()

        # Calculate original C
        dA_L = power(self.L, self.dA)
        I = torch.eye(self.dA.size(-1)).to(dA_L)
        C = _conj(_r2c(self.C))  # (H C N)

        dC = torch.linalg.solve(
            I - dA_L.transpose(-1, -2),
            C.unsqueeze(-1),
        ).squeeze(-1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == "linear":
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2 * self.dC[:, :, : self.N]
        elif mode == "diagonal":
            # Eigendecomposition of the A matrix
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print("Diagonalization error:", torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract("h n m, h m -> h n", V_inv, self.dB)
            self.dC = contract("h n m, c h n -> c h m", V, self.dC)

        elif mode == "dense":
            pass
        else:
            raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        if self._step_mode != "linear":
            N *= 2

            if self._step_mode == "diagonal":
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N),  # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N),  # self.dC.shape
            batch_shape + (H, N),
        )

        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """Must have called self.setup_step() and created state with self.default_state() before calling this"""

        if self._step_mode == "linear":
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y, new_state

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)
