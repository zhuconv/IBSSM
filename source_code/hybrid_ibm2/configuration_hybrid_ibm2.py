import math
import warnings

from transformers.configuration_utils import PretrainedConfig


class HybridIBM2Config(PretrainedConfig):
    model_type = "hybrid_ibm2"

    def __init__(
        self,
        ib_type: str = 'bernoulli',
        return_attn: bool = False,
        auxiliary_loss_weight: float = 0.1,
        max_seq_length: int = 2048,
        num_heads: int = 64,
        head_dim: int = 64,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        state_size: int = 128,
        num_hidden_layers: int = 48,
        layer_norm_epsilon: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        n_groups: int = 1,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        initializer_range: float = 0.1,
        residual_in_fp32: bool = True,
        time_step_rank: str = "auto",
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_floor: float = 1e-4,
        time_step_limit=(0.0, float("inf")),
        rescale_prenorm_residual: bool = True,
        use_cache: bool = True,
        rms_norm: bool = True,
        chunk_size: int = 256,
        attn: dict | None = {
            'layers': (1, 3, 5, 7, 9, 11, 13, 15, 17, 19),
            'num_heads': 16,
            'num_kv_heads': 16,
            'qkv_bias': False,
            'window_size': 2048,
            'rope_theta': 10000.,
        },
        hidden_ratio: int | None = 4,
        hidden_act_mlp: str = "swish",
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        assert ib_type in ['bernoulli', 'gamma'], "Only bernoulli and gamma IB are supported currently."
        self.ib_type = ib_type
        self.return_attn = return_attn
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = (
            math.ceil(self.hidden_size / 16)
            if time_step_rank == "auto"
            else time_step_rank
        )
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.n_groups = n_groups
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rms_norm = rms_norm
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        self.attn = attn
        self.hidden_ratio = hidden_ratio
        self.hidden_act_mlp = hidden_act_mlp
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.tie_word_embeddings = tie_word_embeddings

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def ib_layers(self):
        """Compute IB layer indices: first, middle, last among recurrence layers."""
        if self.attn is None:
            return [0, self.num_hidden_layers // 2 - 1, self.num_hidden_layers - 1]
        recurrence_layers = [i for i in range(self.num_hidden_layers) if i not in self.attn['layers']]
        n = len(recurrence_layers)
        if n == 0:
            return []
        if n == 1:
            return [recurrence_layers[0]]
        if n == 2:
            return [recurrence_layers[0], recurrence_layers[-1]]
        return [recurrence_layers[0], recurrence_layers[n // 2 - 1], recurrence_layers[-1]]
