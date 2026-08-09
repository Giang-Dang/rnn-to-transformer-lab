"""The 1997 memory cell, and the two variants that replaced it.

Hochreiter and Schmidhuber's cell is built around one idea: make the
derivative of the state with respect to its own previous value equal to 1 by
construction, instead of hoping a weight matrix keeps it near 1. Everything
else in the architecture is there to make that idea survive contact with a
network that also has to read and write the cell.

Three things in here are easy to get wrong from memory, so all three are
written the way the paper writes them:

* **There is no forget gate.** The 1997 state update is

      c_t = c_{t-1} + i_t g_in(z_c(t))

  with a self-connection fixed at 1.0. The forget gate is Gers, Schmidhuber
  and Cummins (2000), three years later, and `LstmForget` below is that
  version rather than this one.
* **The squashing functions are not tanh.** The paper's appendix A.1 gives
  g with range [-2, 2] and h with range [-1, 1], both built from the logistic
  sigmoid rather than from tanh. The book renames them g_in and g_out, because
  the paper's `h` collides with the book's hidden state h_t; appendix A records
  the collision.
* **The gates read the cell output, not the cell state.** Peephole
  connections, which let a gate see c_t directly, are Gers, Schraudolph and
  Schmidhuber (2002).

Topology. The 1997 paper's hidden layer is fully connected: a gate receives
connections from every memory cell *and* every other gate unit. What the field
converged on instead, and what `torch.nn.LSTM` implements, is a layer form
where all three blocks read the same h_{t-1}. This module implements the layer
form and says so, because that is the thing a reader will meet; `topologies`
below counts the parameters of both, since the two counts are the difference
between the paper's own "factor of 3^2" and the factor of 4 usually quoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def g_in(x: torch.Tensor) -> torch.Tensor:
    """The paper's g, equation (5) of appendix A.1. Range [-2, 2]."""
    return 4.0 * torch.sigmoid(x) - 2.0


def g_in_prime(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return 4.0 * s * (1.0 - s)


def g_out(x: torch.Tensor) -> torch.Tensor:
    """The paper's h, equation (4) of appendix A.1. Range [-1, 1]."""
    return 2.0 * torch.sigmoid(x) - 1.0


def g_out_prime(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return 2.0 * s * (1.0 - s)


def sigmoid_prime(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s)


@dataclass
class LstmState:
    """One time step of the cell, with everything a derivative needs.

    Kept as a record rather than recomputed, because the analytic Jacobians
    below evaluate at the states the run actually saw, and recomputing them
    from the weights is how an off-by-one gets in.
    """

    c: torch.Tensor
    h: torch.Tensor
    i: torch.Tensor
    o: torch.Tensor
    z_i: torch.Tensor
    z_o: torch.Tensor
    z_c: torch.Tensor


@dataclass
class Lstm1997:
    """c_t = c_{t-1} + i_t g_in(z_c), h_t = o_t g_out(c_t). No forget gate.

    A dataclass of plain tensors, matching `PlainRNN`: chapter 4 reaches in and
    sets weights by hand as often as it trains them, and an nn.Module would add
    a registry that most of these experiments do not use. Pass
    `requires_grad=True` tensors when autograd is wanted.
    """

    w_xi: torch.Tensor
    w_hi: torch.Tensor
    b_i: torch.Tensor
    w_xo: torch.Tensor
    w_ho: torch.Tensor
    b_o: torch.Tensor
    w_xc: torch.Tensor
    w_hc: torch.Tensor
    b_c: torch.Tensor

    @property
    def n_hidden(self) -> int:
        return self.w_hi.shape[0]

    @property
    def n_input(self) -> int:
        return self.w_xi.shape[1]

    def parameters(self) -> list[torch.Tensor]:
        return [
            self.w_xi, self.w_hi, self.b_i,
            self.w_xo, self.w_ho, self.b_o,
            self.w_xc, self.w_hc, self.b_c,
        ]

    def step(
        self, c_prev: torch.Tensor, h_prev: torch.Tensor, x: torch.Tensor
    ) -> LstmState:
        """One step.

        Written as `x @ W.T` rather than `W @ x` so that the same code runs a
        single sequence, with x of shape (n_input,), and a batch, with x of
        shape (batch, n_input). The weight shapes are (n_hidden, n_input)
        either way, which is the layout the Jacobian helpers below assume.
        """
        z_i = x @ self.w_xi.T + h_prev @ self.w_hi.T + self.b_i
        z_o = x @ self.w_xo.T + h_prev @ self.w_ho.T + self.b_o
        z_c = x @ self.w_xc.T + h_prev @ self.w_hc.T + self.b_c
        i = torch.sigmoid(z_i)
        o = torch.sigmoid(z_o)
        c = c_prev + i * g_in(z_c)
        h = o * g_out(c)
        return LstmState(c=c, h=h, i=i, o=o, z_i=z_i, z_o=z_o, z_c=z_c)

    def unroll(self, inputs: torch.Tensor, truncate: bool = False) -> list[LstmState]:
        """Run over `inputs` of shape (n_steps, n_input), from a zero state.

        Returns a list of length n_steps + 1 whose entry t is the state at time
        t, so entry 0 is the initial state. The paper sets s_c(0) = 0 and this
        follows it. Indexing matches `PlainRNN.unroll`, and the Jacobian
        helpers below rely on that.

        `truncate=True` is the paper's learning rule rather than a shortcut.
        Detaching h_{t-1} where the three net inputs are formed is exactly the
        three `~_tr 0` substitutions of appendix A.1: an error arriving at
        z_i, z_o or z_c still reaches the weights on those connections, because
        the product with h_{t-1} is still taken, but it is not carried further
        back through h. What is deliberately not detached is c_{t-1}: that path
        is the carousel, and it is the one thing the truncation must leave
        alone.
        """
        dtype = self.w_hi.dtype
        shape = (*inputs.shape[1:-1], self.n_hidden)
        zero = torch.zeros(shape, dtype=dtype)
        states = [
            LstmState(c=zero, h=zero, i=zero, o=zero, z_i=zero, z_o=zero, z_c=zero)
        ]
        for t in range(inputs.shape[0]):
            h_prev = states[-1].h.detach() if truncate else states[-1].h
            states.append(self.step(states[-1].c, h_prev, inputs[t]))
        return states

    def cec_jacobian_truncated(self) -> torch.Tensor:
        """d c_t / d c_{t-1} under the paper's truncation. Exactly the identity.

        Truncated backprop replaces d z_i(t) / d h(t-1), d z_o(t) / d h(t-1)
        and d z_c(t) / d h(t-1) by zero (appendix A.1, the three `~_tr 0`
        lines). With those gone, the only surviving path from c_{t-1} to c_t in

            c_t = c_{t-1} + i_t g_in(z_c(t))

        is the first term, whose derivative is 1. That is the paper's equation
        (30), and it is what "constant error carousel" means: not that the
        derivative is near 1, but that it is 1, for every t, at every distance.

        Takes no state argument on purpose. There is nothing to evaluate.
        """
        return torch.eye(self.n_hidden, dtype=self.w_hi.dtype)

    def cec_jacobian_full(
        self, previous: LstmState, current: LstmState
    ) -> torch.Tensor:
        """d c_t / d c_{t-1} with the truncated paths put back.

        Without truncation there is a second route from c_{t-1} to c_t: it
        leaves the cell through the output g_out, is scaled by the output gate,
        and re-enters through the input gate and the cell input. Writing
        h_{t-1} = o_{t-1} g_out(c_{t-1}), whose derivative in c_{t-1} is
        diagonal,

            d c_t / d c_{t-1}
              = I + [ diag(g_in(z_c)) diag(f'(z_i)) W_hi
                    + diag(i_t) diag(g_in'(z_c)) W_hc ] D,

            D = diag( o_{t-1} * g_out'(c_{t-1}) ).

        `previous` is the state at t-1 and `current` the state at t. The
        o_{t-1} in D is the previous step's output gate, which is a function of
        h_{t-2} and not of c_{t-1}; getting that wrong adds a term that is not
        there.
        """
        d = torch.diag(previous.o * g_out_prime(previous.c))
        leaving = (
            torch.diag(g_in(current.z_c)) @ torch.diag(sigmoid_prime(current.z_i)) @ self.w_hi
            + torch.diag(current.i) @ torch.diag(g_in_prime(current.z_c)) @ self.w_hc
        )
        eye = torch.eye(self.n_hidden, dtype=self.w_hi.dtype)
        return eye + leaving @ d


@dataclass
class LstmForget(Lstm1997):
    """Gers, Schmidhuber and Cummins (2000): the self-connection is learned.

    c_t = f_t c_{t-1} + i_t g_in(z_c), with f_t a fourth gate. Subclassing
    rather than copying, because the point of the bridge box is that the 2000
    cell is the 1997 cell with one gate added and the fixed 1.0 replaced.

    What it costs: d c_t / d c_{t-1} is now diag(f_t) rather than the identity,
    so the carousel is only constant while the forget gate stays open. What it
    buys: a cell that can be reset, which is what a continual stream with no
    sequence boundaries needs.
    """

    w_xf: torch.Tensor = None  # type: ignore[assignment]
    w_hf: torch.Tensor = None  # type: ignore[assignment]
    b_f: torch.Tensor = None  # type: ignore[assignment]

    def parameters(self) -> list[torch.Tensor]:
        return super().parameters() + [self.w_xf, self.w_hf, self.b_f]

    def step(
        self, c_prev: torch.Tensor, h_prev: torch.Tensor, x: torch.Tensor
    ) -> LstmState:
        z_i = x @ self.w_xi.T + h_prev @ self.w_hi.T + self.b_i
        z_o = x @ self.w_xo.T + h_prev @ self.w_ho.T + self.b_o
        z_c = x @ self.w_xc.T + h_prev @ self.w_hc.T + self.b_c
        z_f = x @ self.w_xf.T + h_prev @ self.w_hf.T + self.b_f
        i = torch.sigmoid(z_i)
        o = torch.sigmoid(z_o)
        f = torch.sigmoid(z_f)
        c = f * c_prev + i * g_in(z_c)
        h = o * g_out(c)
        state = LstmState(c=c, h=h, i=i, o=o, z_i=z_i, z_o=z_o, z_c=z_c)
        state.f = f  # type: ignore[attr-defined]
        return state

    def cec_jacobian_truncated(self, forget: torch.Tensor | None = None) -> torch.Tensor:
        """diag(f_t), not the identity. Constant only while the gate is open."""
        if forget is None:
            raise ValueError("the forget-gate Jacobian needs f_t; pass state.f")
        return torch.diag(forget)


def random_lstm(
    n_hidden: int,
    n_input: int,
    generator: torch.Generator,
    scale: float = 0.1,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> Lstm1997:
    """Weights in [-scale, scale], the interval the paper's experiments use.

    The paper initializes in [-0.1, 0.1] for Experiments 3 to 6 and in
    [-0.2, 0.2] for Experiments 1 and 2. Biases start at zero here; the two
    biases the paper does set deliberately, the negative input gate bias
    against state drift and the negative output gate bias against the abuse
    problem, are set by the experiment that studies them rather than hidden in
    a constructor default.
    """

    def block(rows: int, cols: int) -> torch.Tensor:
        raw = torch.rand(rows, cols, generator=generator, dtype=dtype)
        t = (raw * 2.0 - 1.0) * scale
        return t.requires_grad_(requires_grad)

    def bias() -> torch.Tensor:
        return torch.zeros(n_hidden, dtype=dtype, requires_grad=requires_grad)

    return Lstm1997(
        w_xi=block(n_hidden, n_input), w_hi=block(n_hidden, n_hidden), b_i=bias(),
        w_xo=block(n_hidden, n_input), w_ho=block(n_hidden, n_hidden), b_o=bias(),
        w_xc=block(n_hidden, n_input), w_hc=block(n_hidden, n_hidden), b_c=bias(),
    )


def layer_parameters(n_hidden: int, n_input: int, blocks: int) -> int:
    """Weights in a layer-form recurrent unit with `blocks` weight blocks.

    One block is (W_x, W_h, b). A plain RNN has one, the 1997 LSTM has three
    (cell input, input gate, output gate), the 2000 LSTM has four.
    """
    return blocks * (n_hidden * n_input + n_hidden * n_hidden + n_hidden)


def fully_connected_parameters(units: int, n_input: int) -> int:
    """Weights when every unit in the hidden layer sees every other one.

    This is the 1997 paper's own topology, and it is why the paper's discussion
    quotes a factor of 3^2 rather than 3: replacing each hidden unit by three
    units triples the number of sources as well as the number of destinations.
    """
    return units * units + units * n_input + units


def gru_parameters(n_hidden: int, n_input: int) -> int:
    """Cho et al. (2014): reset gate, update gate, candidate. Three blocks.

    The same block count as the 1997 LSTM and one fewer than the 2000 LSTM,
    which is the arithmetic behind the paper calling its unit "much simpler to
    compute and implement".
    """
    return layer_parameters(n_hidden, n_input, blocks=3)
