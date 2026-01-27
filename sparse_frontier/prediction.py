import json
import os
import secrets
import torch.multiprocessing as mp
from typing import Any, Optional
from time import sleep, time

import psutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from vllm.utils import get_open_port

from sparse_frontier.utils.data import load_data_without_predictions
from sparse_frontier.utils.general import save_config
from sparse_frontier.utils.sparsity_server import (
    configure_sparsity_env,
    fetch_sparsity_and_reset,
    start_sparsity_server,
    SparsityServerHandle,
)

COLLECT_SPARSITY = False
MODEL = None


def process_sample(
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Process a single sample through the model.
    
    Args:
        sample: Dictionary containing input text and metadata
        
    Returns:
        Sample dictionary augmented with model prediction
    """
    global MODEL, COLLECT_SPARSITY
    output = MODEL.generate(sample['input_text'])

    output_dict = {
        'pred': output['text'],
        'output_tokens_len': output['output_tokens_len'],
        'index': sample['index'],
    }

    if COLLECT_SPARSITY:
        stats = fetch_sparsity_and_reset()
        prefill_sparsity = None if stats is None else stats.get("prefill_sparsity")
        decode_access_sum = 0.0 if stats is None else float(stats.get("decode_access_sum", 0.0))
        decode_dense_sum = 0.0 if stats is None else float(stats.get("decode_dense_sum", 0.0))

        decode_sparsity = None
        if decode_dense_sum > 0.0:
            decode_sparsity = 1.0 - (decode_access_sum / decode_dense_sum)

        output_dict["prefill_sparsity"] = prefill_sparsity
        output_dict["decode_sparsity"] = decode_sparsity

    return output_dict


def get_free_ports(n: int) -> list[int]:
    """Find N free ports on the local machine."""
    free_ports = []
    sockets = []
    import socket
    
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('localhost', 0))  # Bind to an available port
            free_ports.append(s.getsockname()[1])  # Get the assigned port number
            sockets.append(s)  # Keep the socket open to reserve the port
    finally:
        # Close all sockets to release the ports
        for s in sockets:
            s.close()
    
    return free_ports


def init_worker(
    cfg: Any,
    worker_ports: list[int],
) -> None:
    import torch.multiprocessing as mp
    
    # Respect pre-set CUDA_VISIBLE_DEVICES if available
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        available_gpus = [int(g) for g in visible.split(',')]
        if len(available_gpus) < cfg.gpus:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES specifies {len(available_gpus)} GPU(s) "
                f"but config requires gpus={cfg.gpus}"
            )
    else:
        available_gpus = list(range(cfg.gpus))

    if len(mp.current_process()._identity) > 0:
        worker_index = mp.current_process()._identity[0] - 1
        # Calculate GPU slice for this worker from available GPUs
        worker_gpus = available_gpus[worker_index * cfg.tp:(worker_index + 1) * cfg.tp]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, worker_gpus))
        # Get pre-allocated port for this worker
        worker_port = worker_ports[worker_index]
    else:
        # Single worker case - use all available GPUs and first port
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
        worker_port = worker_ports[0]

    # Configure VLLM environment
    os.environ['VLLM_HOST_IP'] = 'localhost'
    os.environ['VLLM_PORT'] = str(worker_port)

    global COLLECT_SPARSITY
    server_handle: Optional[SparsityServerHandle] = None
    COLLECT_SPARSITY = bool(getattr(cfg, "use_attention_patch", False))

    if COLLECT_SPARSITY:
        server_handle = start_sparsity_server(
            host="127.0.0.1",
            port=get_open_port(),
            authkey=secrets.token_hex(16),
        )
        configure_sparsity_env(server_handle.config)

    from sparse_frontier.modelling.models.vllm_model import VLLMModel
    global MODEL
    MODEL = VLLMModel(
        model_path=cfg.model.path,
        max_input_tokens=cfg.max_input_tokens,
        max_output_tokens=cfg.max_output_tokens,
        tensor_parallel_size=cfg.tp,
        seed=cfg.random_seed,
        enable_thinking=cfg.thinking,
    )
    

def predict_task(cfg) -> None:
    import json as _json
    attn_args_plain = cfg.attention.args or {}

    # Set env for sparse attention initialization in all (spawned) processes
    if cfg.use_attention_patch:
        os.environ["SF_USE_ATTENTION_PATCH"] = "1"
        os.environ['SF_ATTENTION_NAME'] = cfg.attention.name
        os.environ['SF_ATTENTION_ARGS_JSON'] = _json.dumps(attn_args_plain)
        os.environ['SF_TP_SIZE'] = str(cfg.tp)
        os.environ['SF_MODEL_NUM_Q_HEADS'] = str(cfg.model.num_q_heads)
        os.environ['SF_MODEL_NUM_KV_HEADS'] = str(cfg.model.num_kv_heads)
        os.environ['SF_MODEL_NUM_LAYERS'] = str(cfg.model.num_layers)
        os.environ['SF_MAX_INPUT_TOKENS'] = str(cfg.max_input_tokens)
        os.environ['SF_MAX_OUTPUT_TOKENS'] = str(cfg.max_output_tokens)
        os.environ['SF_KV_CACHE_BLOCK_SIZE'] = str(cfg.kv_cache_block_size)
    else:
        # Explicitly disable the attention patch (plugin defaults to enabled)
        os.environ["SF_USE_ATTENTION_PATCH"] = "0"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    pred_path = cfg.runtime.pred_path
    data = load_data_without_predictions(cfg)
    num_workers = cfg.gpus // cfg.tp
    worker_ports = get_free_ports(num_workers)

    if num_workers == 1:
        init_worker(cfg, worker_ports)
        with open(pred_path, 'at', encoding="utf-8", buffering=1) as fout:
            for sample in tqdm(data, total=len(data)):
                sample_results = process_sample(sample)
                fout.write(json.dumps(sample_results) + '\n')
                fout.flush()

        global MODEL
        del MODEL
        MODEL = None
        
        import gc
        gc.collect()
        
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
    else:
        # Multi-worker case - process samples in parallel
        executor = ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(cfg, worker_ports), mp_context=mp.get_context('spawn'))
        with open(pred_path, 'at', encoding="utf-8", buffering=1) as fout:
            futures = {executor.submit(process_sample, sample): sample for sample in data}
            for future in tqdm(as_completed(futures), total=len(data)):
                sample_results = future.result()
                fout.write(json.dumps(sample_results) + '\n')
                fout.flush()
        
        worker_pids = [p.pid for p in executor._processes.values() if p.is_alive()]
        
        for pid in worker_pids:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass

        executor.shutdown()
        sleep(0.5)

    save_config(os.path.dirname(pred_path), cfg)
    print(f'Prediction for task {cfg.task.name} is done. Output is saved to {pred_path}.')
    