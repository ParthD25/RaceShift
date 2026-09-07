# Data and Model Findings

## Formula 1 data

### FastF1
Primary reproducible F1 Python source for lap timing, tyre state, session data, weather and telemetry workflows.

Repository: https://github.com/theOehrly/Fast-F1

### OpenF1
Useful for historical API access, recent timing, car data, stints, weather, position, intervals and race control.

Documentation: https://openf1.org/docs/

Historical use is the default RaceShift integration. Real-time access is optional and should never be required for the portfolio demo.

### F1 StratLab Strategy Dataset
Hugging Face dataset for strategy/lap prototyping.

https://huggingface.co/datasets/VforVitorio/f1-strategy-dataset

### F1 Corner Telemetry 2024-2025
Large Hugging Face corner-level telemetry source.

https://huggingface.co/datasets/FlorindoDev/f1_corner_telemetry_2024_2025

### Renumics F1 dataset
Small telemetry dataset suitable for fast prototyping.

https://huggingface.co/datasets/renumics/f1_dataset

## Endurance data

### IMSA
Useful later for multi-class endurance laps, stints, sectors and weather.

https://huggingface.co/datasets/tobil/imsa

### FIA WEC / Le Mans
Use user-acquired timing data through a separate adapter. Do not redistribute rights-sensitive raw timing archives without checking source terms.

## Hugging Face model findings

Models investigated for **frozen zero-shot comparison** include time-series foundation models such as:

- Amazon Chronos family / Chronos-2
- IBM Granite Tiny Time Mixer
- Datadog Toto
- TimesFM
- Moirai
- MOMENT
- Lag-Llama
- TiRex family

The project does not fine-tune these models in its primary path because standard fine-tuning would violate the no-global-backprop requirement.

General LLMs such as Qwen and Kimi are not the numerical pace model. They solve a different modality. A small LLM could later explain already-computed model findings, but it must not generate the underlying pace forecast.

## Kaggle policy

Kaggle can be useful for weather/tyre benchmark datasets and exploratory notebooks. Every imported Kaggle dataset must have its license and provenance documented before inclusion in a public repository.

## Data packaging policy

The Git repository contains acquisition code and manifests, not multi-gigabyte raw datasets. Keep raw data under `data/raw/`, `data/cache/` or mounted Google Drive storage and exclude it from Git.
