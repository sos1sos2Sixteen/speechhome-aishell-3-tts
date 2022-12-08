
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math
from .layers import LinearNorm
from typing import Tuple, Optional
from torch import Tensor

GMMParameter = Tuple[Tensor, Tensor, Tensor, Tensor]

def gmm_parameter_from_mlp_v2(mlp_vector: Tensor) -> GMMParameter : 
    # input `mlp vector` should be of shape (bcsz, 3*K) for w_hat, delta_hat and sigma_hat
    # get respective components by slicing
    # mlp_vector : (bcsz, 3 * k)

    bcsz, _ = mlp_vector.size()

    mlp_vector = mlp_vector.view(bcsz, 3, -1)

    w_hat     = mlp_vector[:, 0, :]
    delta_hat = mlp_vector[:, 1, :]
    sigma_hat = mlp_vector[:, 2, :]


    Sigma = F.softplus(sigma_hat)
    Delta = F.softplus(delta_hat)
    W     = F.softmax(w_hat, dim=1)
    Z     = 2 * math.pi * Sigma * Sigma

    return (Z, W, Delta, Sigma)

class Attention(nn.Module): 
    """
    GMMv2 attention mechanism from 
    [E. Battenberg et.al. location-relative attention mechanism for robust long-form speech synthesis]
    """

    def __init__(self, 
        input_size: int, 
        memory_dim: int,
        attention_dim: int,
        k: int, 
        use_last_context: bool
    ) -> None: 
        super().__init__()

        self.K = k
        self.use_last_context = use_last_context
        self.score_mask_value = -1e8

        if self.use_last_context: 
            input_size += memory_dim
        
        # (bcsz, attn_dim) -> (bcsz, 3 * k)
        self.weight_MLP = nn.Sequential(
            LinearNorm(
                input_size, attention_dim, 
                True, 'tanh'
            ), 
            nn.Tanh(), 
            LinearNorm(
                attention_dim, 
                self.K * 3, 
                False, 'linear'
            )
        )

        # attention state 
        self.last_mu = None
        self.context_vec = None



    # fake memory-layer
    def memory_layer(self, x: Tensor) -> Tensor: 
        # this call is also a cue for clearing states for a new batch of data
        self.last_mu, self.context_vec = None, None
        return x
    
    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask: Optional[Tensor]):
        """
        PARAMS
        ------
        attention_hidden_state: (bcsz, h): attention rnn last output
        memory: (bcsz, Tt, f): encoder outputs
        processed_memory: (bcsz, Tt, f'): processed encoder outputs
        attention_weights_cat: (bcsz, 2, Tt): previous and cummulative attention weights
        mask: (bcsz, Tt): binary mask for padded data
        """

        '''
        1. read stete values: 
        S_i         <- attention_hidden_state
        last_mu     <- state.last_mu
        last_ctx    <- state.last_ctx
        '''

        s_i = attention_hidden_state    # (bcsz, h)

        # possible init
        if self.last_mu is None: 
            self.last_mu = torch.zeros(s_i.size(0), self.K, device=s_i.device)
        last_mu = self.last_mu          # (bcsz, k)

        # possible init
        if self.context_vec is None: 
            self.context_vec = torch.zeros(s_i.size(0), memory.size(-1), device=s_i.device)
        last_ctx = self.context_vec     # (bcsz, memory_dim)

        '''
        2. calculate weights
        Z, W, delta, sigma  <- MLP(s_i)
        new_mu              <- last_mu + delta
        energy              <- GMM(Z, W, new_mu, sigma)
        '''

        if self.use_last_context: 
            # (bcsz, h + memory_dim)
            mlp_input = torch.cat((s_i, last_ctx), dim=1)
        else: 
            # (bcsz, h)
            mlp_input = s_i
        
        # each (bcsz, k)
        Z, W, delta, sigma = gmm_parameter_from_mlp_v2(self.weight_MLP(mlp_input))
        new_mu = last_mu + delta

        _, Tt, _ = memory.shape

        # (bcsz, Tt)
        energy = self.energy_from_parameters((Z, W, new_mu, sigma), Tt)
        if mask is not None: 
            energy = torch.masked_fill(energy, mask, self.score_mask_value)
        # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        energy = F.softmax(energy, dim=1)                                             # normal smax
        # energy = F.softmax(energy - energy.max(dim=0).value, dim=1)                 # exp-trick
        # energy = energy / energy.sum(1, keepdim=True)                               # linear norm

        '''
        3. calculate next context vector
        next_ctx        <- energy x memory
        '''
        # (bcsz, 1, Tt) x (bcsz, Tt, memory_dim) -> (bcsz, 1, memory_dim) -> (bcsz, memdim)
        next_ctx = torch.bmm(
            energy.unsqueeze(1), memory
        ).squeeze(1)

        '''
        4. update states
        last_mu         <- next_mu
        last_ctx        <- next_ctx
        '''
        self.last_mu = new_mu
        self.context_vec = next_ctx

        # 5. returns (bcsz, memory_dim), (bcsz, Tt)
        return next_ctx, energy
    
    def energy_from_parameters(self, gmm_params: GMMParameter, Tmax: int) -> Tensor: 
        '''
        (Z, W, mu, sigma) each: (bcsz, k)
        '''

        # each (bcsz, k, 1)
        Z, W, mu, sigma = (x.unsqueeze(-1) for x in gmm_params)
        
        bcsz, k, _ = Z.shape

        # J: (Tmax, )
        J = torch.arange(0, Tmax, device=Z.device).float() + 0.5
        # J: (Tmax, ) -> (1, 1, Tmax) -> (bcsz, k, Tmax)
        J = J[None, None, ...].repeat(bcsz, k, 1)

        # (bcsz, k, T)
        loggaussian = \
            -0.5 * torch.log(Z) - ((J-mu)**2)/(2 * sigma**2)

        # logw: (bcsz, k, 1)
        logw = torch.log(W)

        '''
        given logPi(x), calculate log mixture probability logSum wi*Pi(x)
        = logSumExpLog wi*Pi(x)
        = logSumExp (Logwi + LogPi(x))
        '''
        # logmp: (bcsz, k, T) -> (bcsz, T)
        log_mixture = torch.logsumexp(loggaussian + logw, dim=1) # mixutre prob
        # log(1e-8) ~= -18
        log_mixture = torch.clamp_min(log_mixture, -18)

        return log_mixture



