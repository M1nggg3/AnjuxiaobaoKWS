from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from wekws.model.kws_model import init_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    model = init_model(configs["model"])
    current = model.state_dict()
    pretrained = torch.load(args.pretrained, map_location="cpu")

    loaded = []
    skipped = []
    for key, value in pretrained.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            current[key] = value
            loaded.append(key)
        else:
            skipped.append({
                "key": key,
                "pretrained_shape": list(value.shape) if hasattr(value, "shape") else None,
                "current_shape": list(current[key].shape) if key in current and hasattr(current[key], "shape") else None,
            })
    model.load_state_dict(current)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    report = {
        "config": args.config,
        "pretrained": args.pretrained,
        "output": str(output),
        "loaded_count": len(loaded),
        "skipped_count": len(skipped),
        "loaded": loaded,
        "skipped": skipped,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
