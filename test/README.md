Purpose: Reproduce a compact, end-to-end pipeline using the sample dataset.

Pipeline flow: build_tokenizer -> training -> likelihood -> instruction -> benchmark

Artifacts:
- `test/json/tokenizers/element` (tokenizer built)
- `test/output/training`
- `test/output/likelihood`
- `test/output/instruction_tuning`


SLURM commands order:
```bash
sbatch test/build_tokenizer.slurm
sbatch test/training.slurm
sbatch test/likelihood.slurm
sbatch test/instruction.slurm
sbatch test/benchmark.slurm
```

Notes:
- The tokenizer created by the build step is small. Prefer `json/tokenizers/element` for production.
