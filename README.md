# Audio-MAE

MLP-based autoencoder for audio. Submission for [HEAR-2021@NeurIPS'21](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html).

## Setup

```
git clone https://github.com/ID56/HEAR-2021-Audio-MAE.git
python3 -m pip install HEAR-2021-Audio-MAE
```
## Usage
The module to be imported after installation is `audiomlp`.

```python
from audiomlp import load_model, get_timestamp_embeddings, get_scene_embeddings

model = load_model("checkpoints/audio-mae-f4-v1.pth") # Or load f8-v2

b, ms, sr = 2, 1000, 16000
dummy_input = torch.randn(b, int(sr * ms / 1000))

embeddings, timestamps = get_timestamp_embeddings(dummy_input, model)
scene_embeddings = get_scene_embeddings(dummy_input, model)
```

The installation can also be verified by the [validator](https://github.com/neuralaudio/hear-validator):

```
hear-validator audiomlp --model checkpoints/<ckpt>.pth
```

---

The model can also be used independent of the HEAR common API:

```python
from audiomlp.models import AudioMAE_Wrapper

model = AudioMAE_Wrapper(
    timestamp_embedding_size=8,
    scene_embedding_size=1584,
    encoder_params={"embed_dim": 8}
)
```

## Models

|   Model Name    | Release   | # Params† | GFLOPS* | Sampling Rate | Hop Length | Timestamp Embedding | Scene Embedding |  Location     |
| --------------- | --------- | --------- | ------- | ------------- | ---------- | ------------------- | --------------- | ------------- |
| audio-mae-f4-v1 | v1.0.0-f4 |    212K   | 0.046   |    16000      |    10ms    |  4                  |   792           |  [audio-mae-f4-v1(1.8Mb)](checkpoints/audio-mae-f4-v1.pth)   |
| audio-mae-f8-v2 | v2.0.0-f8 |    213K   | 0.046   |    16000      |    10ms    |  8                  |   1584          |  [audio-mae-f8-v2(1.8Mb)](checkpoints/audio-mae-f8-v2.pth)   |

† <sub>Only considering the encoder, which is used for generating embeddings. The whole autoencoder has twice the parameters.</sub><br>
\* <sub>Although there is no direct way to count FLOPS like parameters, you can use [facebookresearch/fvcore](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md). The FLOPS measured are per single 1s audio input (tensor of shape `(1, 16000)`).</sub>

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/80x15.png" /></a><br />The trained checkpoints are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>, as per HEAR-2021 requirements. You may also download them from drive: [ [audio-mae-f4-v1](https://drive.google.com/uc?id=1Fw60-jSVDMabhKaZIqnzAYHYZoNyRX8s&export=download) | [audio-mae-f8-v2](https://drive.google.com/uc?id=14p4i3JkE-OFtv43OiWS43hsoTACZhOBd&export=download) ].

## Notes

All models were trained on:
- A standard Kaggle environment: a single 16GiB NVIDIA Tesla P100, CUDA 11.0, CuDNN 8.0.5.
- Training splits from the three open tasks.
    - For speech commands, all the 84K, 1s training samples were used
    - For nsynth, random 1s crops were taken from each of the 40K, 4s training samples
    - For dcase, 1s crops were taken at all the annotated events in the 44, 120s training samples
    - In total, there are about 127000 training samples, which equates to roughly 35 hours of training audio.