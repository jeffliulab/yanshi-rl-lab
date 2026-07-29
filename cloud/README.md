# Cloud / HPC training

Templates for running training away from a local workstation:

- `Dockerfile` — headless training image recipe (build it yourself with
  `ACCEPT_EULA=Y`; we intentionally do **not** distribute prebuilt images,
  because Omniverse Kit components are not redistributable).
- `slurm/train.sbatch` — Slurm submission template for university clusters
  (Apptainer/enroot container route).

> ⚠️ Honesty note: these templates are maintained against documentation and
> community-verified recipes, but this project is developed on a single local
> workstation — they have **not** been end-to-end verified on a real cluster
> yet. Treat them as a starting point and run `yanshi doctor` on the target
> machine first. SkyPilot one-command launch is on the backlog.
