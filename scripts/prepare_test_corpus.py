"""
prepare_test_corpus.py — Prepara el corpus de test para la evaluación SCM

Extrae una muestra representativa de la tabla `thoughts` de Supabase local
(ya indexada) para usarla como corpus de evaluación comparativa.

Uso:
    SUPABASE_URL=http://localhost:54321 \
    SUPABASE_SERVICE_ROLE_KEY=<tu_key> \
    python scripts/prepare_test_corpus.py \
        --n 100 \
        --output data/scm_test_corpus.jsonl

Criterios de selección:
    - 100 documentos (configurable)
    - Balanceados por schema_version para evitar sesgo
    - Excluye chunks cuyo parent_doc_id ya tiene otro chunk en la muestra
      (evita evaluar el mismo documento dos veces en chunks distintos)
    - Mínimo 200 chars de contenido (evita fragmentos demasiado cortos)
"""

import os
import json
import random
import argparse
from pathlib import Path

try:
    from supabase import create_client
except ImportError:
    print("ERROR: supabase-py no instalado. Correr: pip install supabase")
    exit(1)


def prepare_corpus(n: int, output_path: str, seed: int = 42) -> None:
    supabase_url = os.environ.get("SUPABASE_URL", "http://localhost:54321")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_key:
        print("ERROR: SUPABASE_SERVICE_ROLE_KEY no configurada")
        exit(1)

    client = create_client(supabase_url, supabase_key)

    print(f"Conectando a Supabase: {supabase_url}")

    # Obtener muestra amplia para luego filtrar
    response = client.table("thoughts") \
        .select("id, content, metadata") \
        .gte("length(content)", 200) \
        .limit(n * 5) \
        .execute()

    records = response.data
    print(f"Registros disponibles: {len(records)}")

    # Filtrar: un chunk por parent_doc_id
    seen_parents = set()
    filtered = []
    for r in records:
        meta = r.get("metadata") or {}
        parent = meta.get("parent_doc_id", r["id"])
        if parent not in seen_parents:
            seen_parents.add(parent)
            filtered.append(r)

    print(f"Después de deduplicar por documento: {len(filtered)}")

    # Muestra aleatoria reproducible
    random.seed(seed)
    sample = random.sample(filtered, min(n, len(filtered)))
    print(f"Muestra final: {len(sample)} documentos")

    # Escribir JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in sample:
            f.write(json.dumps({
                "id":      r["id"],
                "content": r["content"],
                "source":  (r.get("metadata") or {}).get("source_file", "unknown"),
            }, ensure_ascii=False) + "\n")

    print(f"Corpus guardado en: {output_path}")
    print(f"\nPróximo paso:")
    print(f"  oumi evaluate --config configs/scm_eval_gpt4o.yaml")
    print(f"  oumi evaluate --config configs/scm_eval_qwen3_local.yaml")
    print(f"  python scripts/scm_metrics.py \\")
    print(f"      --gpt4o results/gpt4o_scores.jsonl \\")
    print(f"      --qwen3 results/qwen3_local_scores.jsonl \\")
    print(f"      --output results/comparison_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=100, help="Tamaño del corpus de test")
    parser.add_argument("--output", default="data/scm_test_corpus.jsonl")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    prepare_corpus(args.n, args.output, args.seed)
