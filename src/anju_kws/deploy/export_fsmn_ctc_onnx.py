import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import yaml

from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx_model", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    model = init_model(configs["model"])
    if configs["training_config"].get("criterion", "max_pooling") == "ctc":
        model.forward = model.forward_softmax
    load_checkpoint(model, args.checkpoint)
    model.eval()

    feature_dim = configs["model"]["input_dim"]
    backbone_conf = configs["model"]["backbone"]
    num_layers = int(backbone_conf["num_layers"])
    cache_dim = int(backbone_conf.get("proj_dim", model.hdim))
    cache_len = int(model.backbone.padding)

    dummy_input = torch.randn(1, 100, feature_dim, dtype=torch.float)
    cache = torch.zeros(1, cache_dim, cache_len, num_layers, dtype=torch.float)

    output_path = Path(args.onnx_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input, cache),
        str(output_path),
        input_names=["input", "cache"],
        output_names=["output", "r_cache"],
        dynamic_axes={"input": {1: "T"}, "output": {1: "T"}},
        opset_version=13,
        verbose=False,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(output_path))
    meta = onnx_model.metadata_props.add()
    meta.key, meta.value = "cache_dim", str(cache_dim)
    meta = onnx_model.metadata_props.add()
    meta.key, meta.value = "cache_len", str(cache_len)
    meta = onnx_model.metadata_props.add()
    meta.key, meta.value = "num_layers", str(num_layers)
    onnx.save(onnx_model, str(output_path))

    torch_output = model(dummy_input, cache)
    ort_sess = ort.InferenceSession(str(output_path))
    ort_output = ort_sess.run(
        None,
        {
            "input": dummy_input.numpy(),
            "cache": cache.numpy(),
        },
    )
    output_ok = torch.allclose(torch_output[0], torch.tensor(ort_output[0]), atol=1e-5)
    cache_ok = torch.allclose(torch_output[1], torch.tensor(ort_output[1]), atol=1e-5)
    print(
        {
            "onnx_model": str(output_path),
            "feature_dim": feature_dim,
            "cache_dim": cache_dim,
            "cache_len": cache_len,
            "num_layers": num_layers,
            "output_close": bool(output_ok),
            "cache_close": bool(cache_ok),
        }
    )
    if not output_ok or not cache_ok:
        raise RuntimeError("ONNX verification failed")


if __name__ == "__main__":
    main()
