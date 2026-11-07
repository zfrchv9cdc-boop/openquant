"""User-facing sentence encoding command."""

import json
from pathlib import Path


DEFAULT_ARCHIVE = (
    "models/minilm/minilm_int4_g32.safetensors"
)


def configure_encode_command(commands):
    parser = commands.add_parser(
        "encode",
        help="Encode sentences using compressed INT4 MiniLM",
    )

    sources = parser.add_mutually_exclusive_group(
        required=True
    )
    sources.add_argument(
        "--text",
        action="append",
        help="Sentence to encode; repeat for multiple sentences",
    )
    sources.add_argument(
        "--input",
        type=Path,
        help="UTF-8 file with one sentence per line",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(DEFAULT_ARCHIVE),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="Enable CUDA Graphs for repeated input shapes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file",
    )


def load_sentences(args):
    if args.text is not None:
        sentences = list(args.text)
    else:
        sentences = args.input.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        sentences = [
            line.strip()
            for line in sentences
            if line.strip()
        ]

    if not sentences or any(
        not sentence.strip()
        for sentence in sentences
    ):
        raise ValueError(
            "Provide at least one nonempty sentence"
        )

    return sentences


def run_encode(args, *, encoder_factory=None):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    if not 4 <= args.max_length <= 512:
        raise ValueError(
            "max-length must be between 4 and 512"
        )

    output = Path(args.output)
    archive = Path(args.archive)
    input_path = args.input

    protected_paths = {archive.resolve()}

    if input_path is not None:
        protected_paths.add(input_path.resolve())

    if output.resolve() in protected_paths:
        raise ValueError(
            "Output must not replace the input or checkpoint"
        )

    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output}. Use --overwrite."
        )

    sentences = load_sentences(args)

    if encoder_factory is None:
        # Import CUDA-only dependencies when actually encoding.
        from openquant.encoder_v07 import OpenQuantEncoder
        encoder_factory = OpenQuantEncoder

    encoder = encoder_factory(
        archive=archive,
        batch_size=args.batch_size,
        max_length=args.max_length,
        enable_graphs=args.graphs,
        graph_capture_min_calls=2,
        graph_capture_max_attempts=8,
    )

    embeddings = encoder.encode(
        sentences,
        use_graph=args.graphs,
        return_cpu=True,
    ).tolist()

    if len(embeddings) != len(sentences):
        raise RuntimeError(
            "Embedding count differs from input count"
        )

    dimension = len(embeddings[0])

    if dimension < 1 or any(
        len(row) != dimension
        for row in embeddings
    ):
        raise RuntimeError(
            "Inconsistent embedding dimensions"
        )

    document = {
        "schema_version": 1,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "precision": "groupwise_int4",
        "checkpoint": str(archive),
        "count": len(sentences),
        "dimensions": dimension,
        "sentences": sentences,
        "embeddings": embeddings,
    }

    serialized = json.dumps(
        document,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_name(
        output.name + ".tmp"
    )

    if temporary.exists():
        raise FileExistsError(
            f"Temporary output already exists: {temporary}"
        )

    try:
        temporary.write_text(
            serialized,
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"Encoded {len(sentences)} sentences "
        f"into {dimension}-dimensional vectors."
    )
    print("Saved:", output.resolve())

    return 0
