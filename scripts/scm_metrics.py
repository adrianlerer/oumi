"""
scm_metrics.py — Métricas personalizadas para evaluación SCM Concept Bottleneck

Calcula:
1. concept_score_variance: varianza promedio de scores entre corridas (confidence_spread)
2. confidence_spread_distribution: distribución de spreads por principio
3. cosine_similarity_vs_gpt4o: similitud vectorial entre Qwen3 y GPT-4o en el mismo texto

Uso con Oumi: referenciado en los archivos YAML de evaluación.
Uso standalone:
    python scripts/scm_metrics.py \
        --gpt4o results/gpt4o_scores.jsonl \
        --qwen3 results/qwen3_local_scores.jsonl \
        --output results/comparison_report.json
"""

import json
import math
import argparse
from pathlib import Path
from typing import Any

PRINCIPLES = [
    "Legalidad", "Igualdad", "DignidadHumana", "JerarquiaNormativa",
    "BuenaFe", "ConfianzaLegitima", "NoDanar", "Proporcionalidad",
    "Equidad", "AbusoDerecho", "Responsabilidad", "NoEnriquecimientoSinCausa",
    "Solidaridad", "DebidoProceso", "PresuncionInocencia", "Irretroactividad",
    "InDubioProHomine", "AutonomiaVoluntad", "PactaSuntServanda", "Integridad",
    "TratoDigno", "Publicidad", "Motivacion", "TutelaJudicialEfectiva"
]

HIGH_SPREAD_THRESHOLD = 0.15
MEDIUM_SPREAD_THRESHOLD = 0.05


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def extract_score_vector(record: dict) -> list[float]:
    scores = record.get("principles_scores", {})
    return [scores.get(p, 0.0) for p in PRINCIPLES]


def compute_spread_stats(records: list[dict]) -> dict[str, Any]:
    """
    Para un conjunto de corridas del mismo texto, calcula el spread por principio.
    Útil cuando se corren múltiples evaluaciones del mismo documento.
    """
    if not records:
        return {}

    principle_values: dict[str, list[float]] = {p: [] for p in PRINCIPLES}
    for r in records:
        scores = r.get("principles_scores", {})
        for p in PRINCIPLES:
            principle_values[p].append(scores.get(p, 0.0))

    stats = {}
    for p, vals in principle_values.items():
        if not vals:
            continue
        spread = max(vals) - min(vals)
        avg = sum(vals) / len(vals)
        stats[p] = {
            "mean":   round(avg, 4),
            "spread": round(spread, 4),
            "stable": spread < MEDIUM_SPREAD_THRESHOLD,
            "flag":   spread > HIGH_SPREAD_THRESHOLD,
        }
    return stats


def compare_models(gpt4o_records: list[dict], qwen3_records: list[dict]) -> dict[str, Any]:
    """
    Compara vectores de scores entre GPT-4o y Qwen3-8B sobre el mismo corpus.
    Produce:
    - cosine_similarity promedio por documento
    - principios con mayor divergencia
    - distribución de similitudes
    """
    if len(gpt4o_records) != len(qwen3_records):
        print(f"AVISO: longitudes distintas ({len(gpt4o_records)} vs {len(qwen3_records)}). "
              f"Usando min({len(gpt4o_records)}, {len(qwen3_records)}) documentos.")

    n = min(len(gpt4o_records), len(qwen3_records))
    similarities = []
    per_principle_diff: dict[str, list[float]] = {p: [] for p in PRINCIPLES}

    for i in range(n):
        v_gpt = extract_score_vector(gpt4o_records[i])
        v_qw3 = extract_score_vector(qwen3_records[i])

        sim = cosine_similarity(v_gpt, v_qw3)
        similarities.append(sim)

        # Diferencia absoluta por principio
        for j, p in enumerate(PRINCIPLES):
            per_principle_diff[p].append(abs(v_gpt[j] - v_qw3[j]))

    # Principios con mayor divergencia promedio
    avg_diff = {p: round(sum(v) / len(v), 4) for p, v in per_principle_diff.items() if v}
    sorted_diff = sorted(avg_diff.items(), key=lambda x: x[1], reverse=True)

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

    # Distribución de similitudes
    buckets = {"> 0.95": 0, "0.85-0.95": 0, "0.70-0.85": 0, "< 0.70": 0}
    for s in similarities:
        if s > 0.95:   buckets["> 0.95"] += 1
        elif s > 0.85: buckets["0.85-0.95"] += 1
        elif s > 0.70: buckets["0.70-0.85"] += 1
        else:          buckets["< 0.70"] += 1

    return {
        "n_documents":              n,
        "avg_cosine_similarity":    round(avg_sim, 4),
        "similarity_distribution":  buckets,
        "top_divergent_principles": [
            {"principle": p, "avg_abs_diff": d}
            for p, d in sorted_diff[:8]
        ],
        "interpretation": (
            "HIGH AGREEMENT (Qwen3-8B viable for batch SCM)"
            if avg_sim > 0.90 else
            "MODERATE AGREEMENT (review divergent principles before batch use)"
            if avg_sim > 0.75 else
            "LOW AGREEMENT (GPT-4o required for reliable SCM scoring)"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="SCM evaluation metrics")
    parser.add_argument("--gpt4o",  required=True, help="Path to GPT-4o results JSONL")
    parser.add_argument("--qwen3",  required=True, help="Path to Qwen3 results JSONL")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    args = parser.parse_args()

    gpt4o_records = load_jsonl(args.gpt4o)
    qwen3_records = load_jsonl(args.qwen3)

    report = {
        "gpt4o_corpus_size":  len(gpt4o_records),
        "qwen3_corpus_size":  len(qwen3_records),
        "model_comparison":   compare_models(gpt4o_records, qwen3_records),
        "gpt4o_spread_stats": compute_spread_stats(gpt4o_records),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== SCM Evaluation Report ===")
    cmp = report["model_comparison"]
    print(f"Documents compared:      {cmp['n_documents']}")
    print(f"Avg cosine similarity:   {cmp['avg_cosine_similarity']}")
    print(f"Interpretation:          {cmp['interpretation']}")
    print(f"\nTop divergent principles:")
    for item in cmp["top_divergent_principles"][:5]:
        print(f"  {item['principle']:<28} diff={item['avg_abs_diff']}")
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
