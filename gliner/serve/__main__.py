"""CLI entry point for GLiNER Ray Serve.

Usage:
    python -m gliner.serve --model urchade/gliner_small-v2.1

This starts a Ray Serve deployment that can be accessed via HTTP at:
    http://localhost:8000/gliner

Or from Python using the GLiNERClient.
"""

import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Start GLiNER Ray Serve deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., urchade/gliner_small-v2.1)",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "fp16", "bfloat16", "bf16"],
        help="Data type for model weights",
    )
    model_group.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["int8", None],
        help="Real quantization to apply. Only 'int8' is accepted; for precision "
        "changes (fp16/bf16) use --dtype instead.",
    )

    limits_group = parser.add_argument_group("Model Limits")
    limits_group.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    limits_group.add_argument(
        "--max-span-width",
        type=int,
        default=12,
        help="Maximum span width for entity detection",
    )
    limits_group.add_argument(
        "--max-labels",
        type=int,
        default=-1,
        help="Maximum number of labels (-1 for unlimited)",
    )

    threshold_group = parser.add_argument_group("Thresholds")
    threshold_group.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="Default confidence threshold for entities",
    )
    threshold_group.add_argument(
        "--default-relation-threshold",
        type=float,
        default=0.5,
        help="Default confidence threshold for relations",
    )

    replica_group = parser.add_argument_group("Replica Configuration")
    replica_group.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help="Number of model replicas",
    )
    replica_group.add_argument(
        "--num-gpus-per-replica",
        type=float,
        default=1.0,
        help="Number of GPUs per replica",
    )
    replica_group.add_argument(
        "--num-cpus-per-replica",
        type=float,
        default=1.0,
        help="Number of CPUs per replica",
    )

    batch_group = parser.add_argument_group("Batching Configuration")
    batch_group.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size for dynamic batching",
    )
    batch_group.add_argument(
        "--batch-wait-timeout-ms",
        type=float,
        default=10.0,
        help="Batch wait timeout in milliseconds",
    )
    batch_group.add_argument(
        "--request-timeout-s",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    batch_group.add_argument(
        "--max-ongoing-requests",
        type=int,
        default=256,
        help="Maximum number of ongoing requests",
    )
    batch_group.add_argument(
        "--queue-capacity",
        type=int,
        default=4096,
        help="Request queue capacity",
    )
    batch_group.add_argument(
        "--precompiled-batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of batch sizes to precompile",
    )

    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--route-prefix",
        type=str,
        default="/gliner",
        help="HTTP route prefix",
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port for Ray Serve",
    )
    server_group.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (default: local)",
    )

    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--tokenizer-threads",
        type=int,
        default=4,
        help="Number of tokenizer threads",
    )
    perf_group.add_argument(
        "--decoding-threads",
        type=int,
        default=4,
        help="Number of decoding threads",
    )
    perf_group.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile precompilation",
    )
    perf_group.add_argument(
        "--enable-sequence-packing",
        action="store_true",
        help="Enable sequence packing for improved throughput",
    )
    perf_group.add_argument(
        "--enable-flashdeberta",
        action="store_true",
        help="Enable FlashDeBERTa for faster inference",
    )
    perf_group.add_argument(
        "--warmup-iterations",
        type=int,
        default=3,
        help="Number of warmup iterations per batch size",
    )

    memory_group = parser.add_argument_group("Memory Configuration")
    memory_group.add_argument(
        "--target-memory-fraction",
        type=float,
        default=0.9,
        help="Target GPU memory fraction (0.0-1.0)",
    )
    memory_group.add_argument(
        "--memory-overhead-factor",
        type=float,
        default=1.3,
        help="Memory overhead factor for safety margin",
    )

    args = parser.parse_args()

    precompiled_sizes = [int(x.strip()) for x in args.precompiled_batch_sizes.split(",")]

    from .config import GLiNERServeConfig  # noqa: PLC0415
    from .server import serve  # noqa: PLC0415

    config = GLiNERServeConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        max_span_width=args.max_span_width,
        max_labels=args.max_labels,
        default_threshold=args.default_threshold,
        default_relation_threshold=args.default_relation_threshold,
        num_replicas=args.num_replicas,
        num_gpus_per_replica=args.num_gpus_per_replica,
        num_cpus_per_replica=args.num_cpus_per_replica,
        max_batch_size=args.max_batch_size,
        batch_wait_timeout_ms=args.batch_wait_timeout_ms,
        request_timeout_s=args.request_timeout_s,
        max_ongoing_requests=args.max_ongoing_requests,
        queue_capacity=args.queue_capacity,
        route_prefix=args.route_prefix,
        tokenizer_threads=args.tokenizer_threads,
        decoding_threads=args.decoding_threads,
        enable_compilation=not args.no_compile,
        enable_sequence_packing=args.enable_sequence_packing,
        enable_flashdeberta=args.enable_flashdeberta,
        precompiled_batch_sizes=precompiled_sizes,
        target_memory_fraction=args.target_memory_fraction,
        memory_overhead_factor=args.memory_overhead_factor,
        warmup_iterations=args.warmup_iterations,
        http_port=args.port,
        ray_address=args.ray_address,
    )

    print("=" * 60)  # noqa: T201
    print("GLiNER Ray Serve Configuration")  # noqa: T201
    print("=" * 60)  # noqa: T201
    print(f"Model: {args.model}")  # noqa: T201
    print(f"Device: {args.device}, dtype: {args.dtype}")  # noqa: T201
    if args.quantization:
        print(f"Quantization: {args.quantization}")  # noqa: T201
    print(f"Max batch size: {args.max_batch_size}")  # noqa: T201
    print(f"Precompiled batch sizes: {precompiled_sizes}")  # noqa: T201
    print(f"Num replicas: {config.num_replicas}")  # noqa: T201
    print(f"Port: {args.port}")  # noqa: T201
    print(f"Route prefix: {args.route_prefix}")  # noqa: T201
    print(f"Compilation: {'enabled' if not args.no_compile else 'disabled'}")  # noqa: T201
    print(f"FlashDeBERTa: {'enabled' if args.enable_flashdeberta else 'disabled'}")  # noqa: T201
    print(f"Sequence packing: {'enabled' if args.enable_sequence_packing else 'disabled'}")  # noqa: T201
    print(f"Target memory fraction: {args.target_memory_fraction}")  # noqa: T201
    print("=" * 60)  # noqa: T201

    serve(config, blocking=True)


if __name__ == "__main__":
    main()
