import onnxruntime
import torch
import ailia


def main():
    lip = torch.rand(1, 1, 88, 88, 100).to(torch.float32)
    lip_len = torch.tensor([lip.shape[-1]]).to(torch.int64)
    spk_emb = torch.rand(1, 256).to(torch.float32)

    model = onnxruntime.InferenceSession("model.onnx")
    mel, _ = model.run(
        None,
        {
            'lip': lip.numpy(),
            'lip_len': lip_len.numpy(),
            'spk_emb': spk_emb.numpy(),
        }
    )

    model = ailia.Net(None, "model.onnx")
    mel, _ = model.run(
        {
            'lip': lip.numpy(),
            'lip_len': lip_len.numpy(),
            'spk_emb': spk_emb.numpy(),
        }
    )
    breakpoint()


if __name__ == "__main__":
    main()