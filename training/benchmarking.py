import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logging import get_logger

LOGGER = get_logger(__name__)


class BenchmarkOutcome:
    def __init__(self, name, status, report_path=None, details=None):
        self.name = name
        self.status = status
        self.report_path = Path(report_path) if report_path else None
        self.details = details or {}

    def to_dict(self):
        payload = {
            "name": self.name,
            "status": self.status,
            "details": self.details,
        }
        if self.report_path is not None:
            payload["report_path"] = str(self.report_path)
        return payload


class HFLocalModelWrapper:
    def __init__(self, model_dir, device="auto", torch_dtype="auto", max_new_tokens=512, temperature=0.0):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")

        LOGGER.info("Loading HF causal LM from %s", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        dtype = self._resolve_dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=dtype)

        resolved_device = self._resolve_device(device)
        self.device = torch.device(resolved_device)
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_new_tokens = self._coerce_int(max_new_tokens, "max_new_tokens")
        self.temperature = self._coerce_float(temperature, "temperature")
        self.do_sample = self.temperature > 0.0

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _resolve_dtype(value):
        if value in (None, "auto"):
            return None
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            attr = value.lower()
            if not attr.startswith("torch."):
                attr = f"torch.{attr}"
            try:
                return getattr(torch, attr.split(".")[-1])
            except AttributeError as exc:  # noqa: BLE001
                raise ValueError(f"Unknown torch dtype '{value}'") from exc
        raise TypeError(f"Unsupported dtype specifier: {value!r}")

    @staticmethod
    def _coerce_int(value, name):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError as exc:  # noqa: BLE001
                raise ValueError(f"{name} must be an integer") from exc
        raise TypeError(f"{name} must be an integer, got {type(value)}")

    @staticmethod
    def _coerce_float(value, name):
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as exc:  # noqa: BLE001
                raise ValueError(f"{name} must be a number") from exc
        raise TypeError(f"{name} must be a number, got {type(value)}")
    @staticmethod
    def _normalise_content(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        value = item.get("text") or item.get("value")
                        if value:
                            parts.append(str(value))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url")
                        if isinstance(url, dict):
                            url = url.get("url")
                        if url:
                            parts.append(f"[image: {url}]")
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    def _format_messages(self, raw_messages):
        messages = raw_messages
        if isinstance(messages, dict) and "messages" in messages:
            messages = messages["messages"]
        if not isinstance(messages, list):
            raise TypeError(f"Unsupported prompt payload: {type(messages)}")

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Falling back to manual prompt formatting: %s", exc)

        formatted = []
        for message in messages:
            role = message.get("role", "user") if isinstance(message, dict) else "user"
            content = message.get("content") if isinstance(message, dict) else message
            formatted_content = self._normalise_content(content)
            formatted.append(f"{role.upper()}: {formatted_content}")
        formatted.append("ASSISTANT:")
        return "\n".join(formatted)

    def _prepare_inputs(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _decode(self, outputs, input_length):
        generated = outputs[0][input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate(self, prompts, **generation_kwargs):
        from chembench.types import Generation, Generations

        responses = []
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if "max_tokens" in generation_kwargs and "max_new_tokens" not in generation_kwargs:
            generation_kwargs = dict(generation_kwargs)
            kwargs["max_new_tokens"] = generation_kwargs.pop("max_tokens")
        kwargs.update(generation_kwargs)

        unused_keys = {
            "model",
            "headers",
            "model_response",
            "print_verbose",
            "acompletion",
            "logging_obj",
            "optional_params",
            "litellm_params",
            "timeout",
            "custom_prompt_dict",
            "encoding",
        }
        for key in list(kwargs):
            if key in unused_keys:
                kwargs.pop(key)

        for prompt in prompts:
            formatted_prompt = self._format_messages(prompt)
            inputs = self._prepare_inputs(formatted_prompt)
            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **kwargs)
            text = self._decode(outputs, input_len)
            responses.append([Generation(text=text)])

        return Generations(generations=responses)


def _latest_summary(report_root):
    report_root = Path(report_root)
    summaries = sorted(report_root.rglob("summary.json"))
    return summaries[-1] if summaries else None


def run_chembench(
    model_wrapper,
    model_name,
    output_dir,
    dataset_name="jablonkagroup/ChemBench",
    prompt_type="instruction",
    topics=None,
):
    try:
        from chembench.evaluate import ChemBenchmark, save_topic_reports
        from chembench.prompter import PrompterBuilder
        from chembench.utils import enable_logging
    except ImportError as exc:  # noqa: BLE001
        return BenchmarkOutcome(
            name="chembench",
            status="missing-dependency",
            details={
                "error": str(exc),
                "hint": "Install chembench (pip install chembench) before running this command.",
            },
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enable_logging()
    run_id = str(model_name).replace("/", "-")
    benchmark = ChemBenchmark.from_huggingface(dataset_name=dataset_name, report_dir=str(output_dir), run_id=run_id)
    prompter = PrompterBuilder.from_model_object(model_wrapper, prompt_type=prompt_type)

    LOGGER.info("Running ChemBench on %s", model_name)
    results = benchmark.bench(prompter, topics=list(topics) if topics else None)
    save_topic_reports(benchmark, results, report_name=run_id)

    summary_path = _latest_summary(output_dir)
    return BenchmarkOutcome(
        name="chembench",
        status="ok",
        report_path=summary_path,
        details={
            "num_results": len(results),
            "report_root": str(output_dir),
        },
    )


def _render_command(template, model_path, output_dir):
    payload = {
        "model": str(model_path),
        "model_dir": str(model_path),
        "output": str(output_dir),
        "output_dir": str(output_dir),
    }
    return template.format(**payload)


def run_external_benchmark(name, command_template, model_path, output_dir, env=None):
    if not command_template:
        return BenchmarkOutcome(
            name=name,
            status="missing-runner",
            details={
                "message": (
                    f"No command template provided for {name}. Use --{name}-command to supply the official runner."
                )
            },
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = _render_command(command_template, model_path=Path(model_path), output_dir=output_dir)
    LOGGER.info("Running %s using command: %s", name, command)
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    log_path = output_dir / "runner.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# Command\n")
        handle.write(command + "\n\n")
        handle.write("# STDOUT\n")
        handle.write(completed.stdout)
        handle.write("\n# STDERR\n")
        handle.write(completed.stderr)
        handle.write(f"\n# EXIT CODE\n{completed.returncode}\n")

    status = "ok" if completed.returncode == 0 else "failed"
    details = {
        "exit_code": completed.returncode,
        "log_file": str(log_path),
    }

    if completed.stdout:
        details["stdout_preview"] = completed.stdout.strip().splitlines()[-5:]
    if completed.stderr:
        details["stderr_preview"] = completed.stderr.strip().splitlines()[-5:]

    return BenchmarkOutcome(name=name, status=status, report_path=output_dir, details=details)


def _sanitise_model_name(path):
    path = Path(path)
    return path.name.replace(" ", "-")


def evaluate_single_model(
    model_path,
    output_root,
    chembench_dataset,
    prompt_type,
    topics,
    device,
    torch_dtype,
    max_new_tokens,
    temperature,
    chempile_command,
    chemiq_command,
):
    model_path = Path(model_path)
    model_name = _sanitise_model_name(model_path)
    model_output_dir = Path(output_root) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = HFLocalModelWrapper(
        model_path,
        device=device,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    outcomes = []
    outcomes.append(
        run_chembench(
            wrapper,
            model_name=model_name,
            output_dir=model_output_dir / "chembench",
            dataset_name=chembench_dataset,
            prompt_type=prompt_type,
            topics=topics,
        )
    )
    outcomes.append(
        run_external_benchmark(
            "chempile",
            chempile_command,
            model_path=model_path,
            output_dir=model_output_dir / "chempile",
        )
    )
    outcomes.append(
        run_external_benchmark(
            "chemiq",
            chemiq_command,
            model_path=model_path,
            output_dir=model_output_dir / "chemiq",
        )
    )

    summary = {
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "outcomes": [outcome.to_dict() for outcome in outcomes],
    }

    summary_path = model_output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def discover_baseline_models(baselines_root, selected=None):
    root = Path(baselines_root)
    if not root.exists():
        raise FileNotFoundError(f"Baseline directory not found: {root}")

    if selected:
        paths = []
        for entry in selected:
            candidate = root / entry
            if candidate.exists():
                paths.append(candidate)
                continue
            alt = root / entry.replace("/", "-")
            if alt.exists():
                paths.append(alt)
                continue
            raise FileNotFoundError(f"Baseline '{entry}' not found under {root}")
        return paths

    return sorted(path for path in root.iterdir() if path.is_dir())


def evaluate_baselines(
    baselines_root,
    model_filters=None,
    output_root=None,
    chembench_dataset="jablonkagroup/ChemBench",
    prompt_type="instruction",
    topics=None,
    device="auto",
    torch_dtype="auto",
    max_new_tokens=512,
    temperature=0.0,
    chempile_command=None,
    chemiq_command=None,
):
    baselines = discover_baseline_models(Path(baselines_root), model_filters)
    if not baselines:
        raise RuntimeError(f"No baseline models found under {baselines_root}")

    output_root = Path(output_root or "artifacts/benchmark-results")
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_dir in baselines:
        LOGGER.info("Evaluating baseline at %s", model_dir)
        summary = evaluate_single_model(
            model_dir,
            output_root=output_root,
            chembench_dataset=chembench_dataset,
            prompt_type=prompt_type,
            topics=topics,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            chempile_command=chempile_command,
            chemiq_command=chemiq_command,
        )
        summaries.append(summary)

    aggregate = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_root": str(output_root),
        "models": [summary["model"] for summary in summaries],
    }
    aggregate_path = output_root / "latest.json"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    LOGGER.info("Wrote aggregate benchmark metadata to %s", aggregate_path)
    return summaries
