from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from project_imports import LOGGER, ROOT_DIR, load_json, save_json

STATE_FILE = ROOT_DIR / "pipeline_state.json"

STAGES = [
    {
        "name": "cortar_batimentos",
        "script": ROOT_DIR / "cortar_batimentos.py",
        "description": "Gera dataset_cortado a partir do CSV principal.",
    },
    {
        "name": "augmentation_ecg",
        "script": ROOT_DIR / "augmentation_ecg.py",
        "description": "Aplica aumento de dados nas classes deficitárias.",
    },
    {
        "name": "split_dataset",
        "script": ROOT_DIR / "split_dataset.py",
        "description": "Divide o dataset em train/val/test sem vazamento.",
    },
    {
        "name": "kolmo_corrigido",
        "script": ROOT_DIR / "kolmo_corrigido.py",
        "description": "Treina o modelo KAN e gera os artefatos as dependências.",
    },
    {
        "name": "benchmark_modelos",
        "script": ROOT_DIR / "benchmark_modelos (3).py",
        "description": "Compara os modelos e gera arquivos de benchmark.",
    },
    {
        "name": "exportar_kan_cpp",
        "script": ROOT_DIR / "exportar_kan_cpp.py",
        "description": "Exporta o modelo KAN para C++/ESP32.",
    },
]


def load_state() -> dict:
    state = load_json(STATE_FILE)
    state.setdefault("completed", {})
    state.setdefault("failed", {})
    state.setdefault("history", [])
    return state


def persist_state(state: dict) -> None:
    save_json(STATE_FILE, state)


def run_stage(stage: dict, state: dict) -> None:
    stage_name = stage["name"]
    script = stage["script"]

    LOGGER.info(f"Executando etapa: {stage_name} -> {stage['description']}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=False,
    )

    if result.returncode == 0:
        state["completed"][stage_name] = True
        state["history"].append({"stage": stage_name, "status": "ok"})
        LOGGER.info(f"Etapa concluída com sucesso: {stage_name}")
    else:
        state["failed"][stage_name] = result.returncode
        state["history"].append({"stage": stage_name, "status": "failed", "code": result.returncode})
        LOGGER.warning(f"Etapa falhou: {stage_name} (código={result.returncode}).")
        LOGGER.warning("A falha ficou registrada no state file. Você pode relançar o pipeline e ele continuará a partir das etapas concluídas.")


def main() -> None:
    state = load_state()
    LOGGER.info(f"Arquivo de checkpoint: {STATE_FILE}")

    for stage in STAGES:
        stage_name = stage["name"]
        if state["completed"].get(stage_name):
            LOGGER.info(f"Pulando etapa já concluída: {stage_name}")
            continue

        run_stage(stage, state)
        persist_state(state)

    LOGGER.info("Pipeline finalizado.")
    LOGGER.info(json.dumps({
        "completed": list(state["completed"].keys()),
        "failed": state["failed"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
