# Audio-MAE

MLP-based autoencoder for audio. Submission for [HEAR-2021@NeurIPS'21](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html).

## Setup

```
git clone https://github.com/ID56/Audio-MLP.git
python3 -m pip install Audio-MLP
```

## Models

|   Model Name    | # Params† | GFLOPS* | Sampling Rate | Hop Length | Timestamp Embedding | Scene Embedding |  Location |
| --------------- | -------- | ------- | ------------- | ---------- | ------------------- | --------------- | ------------- |
| audiomlp-v1.0.0 |   212K   | 0.046   |    16000      |    10ms    |  4                  |   792           |  [audiomlp-v1.0.0(1.8Mb)](checkpoints/audio-mae-f4-v1.pth)   |
| audiomlp-v2.0.0 |   213K   | 0.046   |    16000      |    10ms    |  8                  |   1584          |  [audiomlp-v2.0.0(1.8Mb)](checkpoints/audio-mae-f8-v2.pth)   |

† <sub>Only considering the encoder, which is used for generating embeddings. The whole autoencoder has twice the parameters.</sub><br>
\* <sub>Although there is no direct way to count FLOPS like parameters, you can use [facebookresearch/fvcore](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md). The FLOPS measured are per single 1s audio input (tensor of shape `(1, 16000)`).</sub>
## Notes

All models were trained on:
- A standard Kaggle environment: a single 16GiB NVIDIA Tesla P100, CUDA 11.0, CuDNN 8.0.5.
- Training splits from the three open tasks.
    - For speech commands, all the 84K, 1s training samples were used
    - For nsynth, random 1s crops were taken from each of the 40K, 4s training samples
    - For dcase, 1s crops were taken at all the annotated events in the 44, 120s training samples
    - In total, there are about 127000 training samples, which equates to roughly 35 hours of training audio.