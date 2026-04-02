from contextlib import contextmanager

try:
    from megatron.core import mpu, parallel_state
except ImportError:
    mpu = None
    parallel_state = None

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


def _unwrap_process_group(group):
    return getattr(group, "group", group)


@contextmanager
def patch_megatron_parallel_groups():
    if mpu is None and parallel_state is None:
        yield
        return

    patched = []
    getter_names = (
        "get_data_parallel_group",
        "get_data_parallel_group_gloo",
        "get_data_parallel_group_with_context_parallel",
        "get_context_parallel_group",
        "get_tensor_model_parallel_group",
        "get_pipeline_model_parallel_group",
        "get_embedding_group",
        "get_position_embedding_group",
        "get_expert_model_parallel_group",
        "get_tensor_and_expert_parallel_group",
        "get_data_modulo_expert_parallel_group",
        "get_amax_reduction_group",
    )

    def patch_module(module):
        if module is None:
            return
        for name in getter_names:
            if not hasattr(module, name):
                continue
            original = getattr(module, name)

            def wrapped(*args, __original=original, **kwargs):
                return _unwrap_process_group(__original(*args, **kwargs))

            patched.append((module, name, original))
            setattr(module, name, wrapped)

    patch_module(mpu)
    patch_module(parallel_state)
    try:
        yield
    finally:
        for module, name, original in reversed(patched):
            setattr(module, name, original)


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        with patch_megatron_parallel_groups():
            yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
