
from pathlib import Path
from huggingface_hub import snapshot_download

from utils.logging import get_logger

LOGGER = get_logger(__name__)

def _resolve_default_model_spec(config):
    model_cfg = config.get("model")
    if model_cfg is None:
        raise ValueError("Config is missing 'model' section with at least a name")

    if isinstance(model_cfg, str):
        if not model_cfg.strip():
            raise ValueError("Config model name is empty")
        return {"name": model_cfg.strip()}

    if isinstance(model_cfg, dict):
        name = model_cfg.get("name")
        if not name:
            raise ValueError("Config model section requires a 'name'")

        spec = {"name": name}
        if model_cfg.get("revision"):
            spec["revision"] = model_cfg["revision"]
        if model_cfg.get("model_kwargs"):
            spec["model_kwargs"] = model_cfg["model_kwargs"]
        if model_cfg.get("tokenizer_kwargs"):
            spec["tokenizer_kwargs"] = model_cfg["tokenizer_kwargs"]
        return spec

    raise ValueError("Unsupported type for config.model; expected dict or string")


def _coerce_model_list(selected):
    if not selected:
        return []

    names = []
    seen = set()
    for raw in selected:
        if raw is None:
            continue
        name = raw.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def download_baseline_models(
    config,
    hf_token,
    output_dir=None,
    only_models=None,
):
    explicit_names = _coerce_model_list(only_models)

    if explicit_names:
        specs = [{"name": name} for name in explicit_names]
    else:
        specs = [_resolve_default_model_spec(config)]

    if not specs:
        raise ValueError("Unable to determine any baseline models to download")

    for spec in specs:
        model_name = spec["name"]
        revision = spec.get("revision")

        if output_dir is not None:
            target_dir = Path(output_dir) / model_name.replace("/", "-")
        else:
            target_dir = Path("artifacts/baselines") / model_name.replace("/", "-")

        LOGGER.info("Ensuring baseline model %s is available at %s", model_name, target_dir)

        snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_dir=target_dir,
            token=hf_token,
            local_dir_use_symlinks=False,
        )

        LOGGER.info("Verified baseline model artifacts for %s", model_name)
