# LOCI vision and hearing playground

Arko Sanyal proposed the XYZ+A representation, chunked subject color, native detail for text, focus blur and faster capture during motion. Codex implemented and measured the synthetic shader and feature path. The local model-routing work uses earlier model research from the models and staging agents.

## Reproduce

Use a Linux desktop with EGL and both GPU drivers, Python and DejaVu Sans Mono. From this directory, create a virtual environment and install the pinned dependencies.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p results
stamp=$(date +%F_%H%M%S)
export OPENBLAS_NUM_THREADS=1
.venv/bin/python benchmark.py --device 0 --expect NVIDIA --output "results/${stamp}_rtx.json"
.venv/bin/python benchmark.py --device 2 --expect BMG --output "results/${stamp}_arc.json"
```

Device indices can change. The expected renderer check prevents accidentally measuring the integrated GPU or a software renderer. Run the commands sequentially to reduce interference. Each device records three runs of 240 frames per mode and resolution, plus 24 warm-up frames per run. The script alternates the test order, timestamps the report in local time, and refuses to overwrite an existing report. Measurements from this workstation were taken with the GPU inference services online and a model download running.

## What the benchmark measures

- The shader renders a moving subject and short labels over a background with attenuated high-frequency detail. The packet carries measured screen position and mean subject RGB, plus the shader's authored depth, light incidence and blur values. Background blur is rendered as reduced high-frequency contrast. Its perceptual quality was not evaluated.
- Full mode reads native RGB. Coarse reads a 320 × 180 preview. Hybrid adds a native-resolution text crop. Adaptive uses the authored motion and label change as a reference scheduler. Observed uses the preview's measured position and classified label to choose whether to read native text and emit a packet. In observed mode the preview is still read on every frame.
- The scene advances on a virtual 120 Hz clock without real-time pacing. Processing time includes rendering, GPU readback, CPU feature extraction and JSON serialization where applicable. It excludes capture from a browser or camera, transport, model inference and display.
- Text is a closed-set comparison against four known label templates. All modes recovered the emitted labels. Image error against the template still distinguishes crop fidelity. Localization error uses native-frame pixel units. The audio check uses authored mono PCM bursts and a 10 ms RMS gate. Both runs detected all three onsets at 0.40, 0.90 and 1.40 seconds.
- The fixture has one identifiable object and a known text rectangle. It does not test general OCR, object detection, inferred depth or lighting, occlusion, musical meaning, or audio/video synchronization under jitter.

## Results on 2026-11-20

The raw reports preserve all 90 case runs. The table reports the median run's p95 frame time and the sum of frames exceeding 10 ms out of 720 measured frames for each row.

| 1080p path | RTX 5060 Ti p95 | Intel Arc B70 p95 | Frames above 10 ms on RTX / Arc |
| --- | ---: | ---: | ---: |
| Full readback | 8.68 ms | 10.24 ms | 82 / 301 |
| Coarse | 0.36 ms | 0.59 ms | 0 / 0 |
| Hybrid | 0.40 ms | 0.71 ms | 0 / 0 |
| Observed adaptive | 0.55 ms | 0.88 ms | 0 / 0 |

At 1080p, hybrid reduced template RMS error from 0.219 to 0.051 by retaining the small text crop. Over two simulated seconds, observed scheduling emitted 140 packets versus 240 fixed-rate packets, with 119 packets during the 120-frame motion phase. It reduced serialized packet bytes by about 41% versus fixed hybrid capture. It reduced GPU readback only about 4% versus fixed hybrid because gating reads the preview every frame.

Use 1080p with a 320 × 180 tracking preview and native text crops as the starting profile. At 1440p, the synthetic text metric was unchanged and the localization error increased in native pixel units because the preview size stayed fixed. Both GPUs had processing headroom for 100 updates per second with this fixture. A paced live stream and a model consuming those packets remain separate measurements.
