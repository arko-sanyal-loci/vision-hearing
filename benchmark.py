"""Synthetic XYZ+A, text fidelity, blur and audio-cue benchmark on EGL GPUs."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SOURCE_HZ = 120
LABEL_RECT = (.72, .85, .105, .026)
LABELS = ("B147", "B174", "B741", "B714")
VERTEX = """#version 330
out vec2 uv;
void main() {
    uv = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(uv * 2. - 1., 0., 1.);
}"""
DOWNSAMPLE = """#version 330
in vec2 uv; out vec4 color;
uniform sampler2D source_image;
void main() { color = texture(source_image, uv); }
"""


def rect(width, height):
    x, y, w, h = LABEL_RECT
    return round(x * width), round(y * height), round(w * width), round(h * height)


def glyph(text):
    image = Image.new("L", (140, 32))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 29)
    ImageDraw.Draw(image).text((70, 15), text, font=font, fill=255, anchor="mm")
    return image


def truth(frame, count):
    fraction = frame / count
    x = .15 + .40 * min(max((fraction - .25) / .5, 0), 1)
    return x, .43, 2.0, 30 + 15 * math.sin(fraction * math.tau), LABELS[(frame // 11) % 4]


def capture(fbo, viewport=None):
    width, height = viewport[2:] if viewport else fbo.size
    raw = fbo.read(viewport=viewport, components=3, alignment=1)
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def decode_label(rgb, templates):
    gray = np.flipud(rgb[:, :, 0])
    samples = np.asarray(Image.fromarray(gray).resize((140, 32))).astype(np.float32)
    distances = {label: np.mean((samples - expected) ** 2) for label, expected in templates.items()}
    closest = min(distances, key=distances.get)
    return closest, math.sqrt(distances[closest]) / 255


def blue_mask(pixels):
    blue = pixels[:,:,2].astype(np.int16)
    red = pixels[:,:,0].astype(np.int16)
    return (blue > 140) & (blue - red > 95)


def blue_center(mask):
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise RuntimeError("Synthetic blue subject was not detected")
    return [(float(columns.mean()) + .5) / mask.shape[1],
            (float(rows.mean()) + .5) / mask.shape[0]]


def run(ctx, width, height, mode, frames, packet_sink=None):
    texture = ctx.texture((width, height), 4)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    source = ctx.framebuffer([texture])
    preview = ctx.simple_framebuffer((320, 180), components=4)
    scene = ctx.program(vertex_shader=VERTEX, fragment_shader=Path(__file__).with_name("scene.frag").read_text())
    down = ctx.program(vertex_shader=VERTEX, fragment_shader=DOWNSAMPLE)
    draw = ctx.vertex_array(scene, [])
    reduce = ctx.vertex_array(down, [])
    images = {label: glyph(label) for label in LABELS}
    templates = {label: 10 + np.asarray(image).astype(np.float32) * .89 for label, image in images.items()}
    sprites = {label: ctx.texture(image.size, 1, image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes())
               for label, image in images.items()}
    scene["blur"].value = .7
    scene["glyphs"].value = down["source_image"].value = 0
    sizes, delays, errors, text_errors = [], [], [], []
    hits = 0
    packet_bytes = 0
    last = -99
    previous = None
    previous_label = None
    previous_observed = None
    previous_observed_label = None
    readback_bytes = 0
    phase_counts = [0, 0, 0]
    for frame in range(-24, frames):
        start = time.perf_counter()
        x, y, z, incidence, label = truth(max(0, frame), frames)
        source.use()
        scene["center"].value = x, y
        scene["depth"].value = z
        scene["incidence"].value = math.radians(incidence)
        sprites[label].use()
        draw.render(vertices=3)
        moved = previous is not None and abs(x - previous) * SOURCE_HZ > .1
        changed_text = label != previous_label
        previous, previous_label = x, label
        if mode == "observed" and frame >= 0:
            preview.use()
            texture.use()
            reduce.render(vertices=3)
            pixels = capture(preview)
            readback_bytes += pixels.nbytes
            xy = blue_center(blue_mask(pixels))
            rx, ry, rw, rh = rect(*preview.size)
            preview_label, _ = decode_label(pixels[ry:ry+rh, rx:rx+rw], templates)
            moved = previous_observed is not None and math.dist(xy, previous_observed) * SOURCE_HZ > .1
            changed_text = preview_label != previous_observed_label
            previous_observed, previous_observed_label = xy, preview_label
        interval = 1 if moved else 10
        selected = frame >= 0 and (mode not in ("adaptive", "observed") or frame - last >= interval or changed_text)
        if selected:
            if mode == "full":
                pixels = capture(source)
                rx, ry, rw, rh = rect(width, height)
                text_image = pixels[ry:ry+rh, rx:rx+rw]
                sizes.append(pixels.nbytes)
            elif mode == "observed":
                text_image = capture(source, rect(width, height))
                sizes.append(text_image.nbytes)
            else:
                preview.use()
                texture.use()
                reduce.render(vertices=3)
                pixels = capture(preview)
                if mode == "coarse":
                    rx, ry, rw, rh = rect(*preview.size)
                    text_image = pixels[ry:ry+rh, rx:rx+rw]
                    sizes.append(pixels.nbytes)
                else:
                    text_image = capture(source, rect(width, height))
                    sizes.append(pixels.nbytes + text_image.nbytes)
            mask = blue_mask(pixels)
            xy = blue_center(mask)
            errors.append(math.hypot(xy[0] - x, xy[1] - y) * width)
            observed, text_error = decode_label(text_image, templates)
            text_errors.append(text_error)
            hits += observed == label
            palette = pixels[mask].mean(axis=0).round().astype(int).tolist()
            packet = {"t": frame / SOURCE_HZ, "object": "blue-disc", "xy": xy,
                      "z_authored": z, "incidence_authored": incidence, "blur_authored": .7,
                      "color_rgb": palette, "text": observed, "moving": moved}
            serialized = json.dumps(packet, separators=(",", ":"))
            packet_bytes += len(serialized.encode()) + 1
            if packet_sink is not None:
                packet_sink.write(serialized + "\n")
            last = frame
            phase_counts[0 if frame < frames / 4 else 1 if frame < 3 * frames / 4 else 2] += 1
        elif mode != "observed" or frame < 0:
            ctx.finish()
        if frame >= 0:
            delays.append(time.perf_counter() - start)
    for item in (draw, reduce, scene, down, source, preview, texture, *sprites.values()):
        item.release()
    return {"resolution": [width, height], "mode": mode, "frames": frames,
            "packets": len(sizes), "packets_by_phase": phase_counts,
            "throughput_fps": frames / sum(delays),
            "frame_ms_p95": 1000 * np.percentile(delays, 95),
            "frames_over_10ms": sum(delay > .01 for delay in delays),
            "gpu_readback_bytes": sum(sizes) + readback_bytes, "packet_json_bytes": packet_bytes,
            "position_error_px_p95": np.percentile(errors, 95),
            "text_candidates_correct": hits, "text_error_rms_median": np.median(text_errors)}


def audio_check():
    rate = 24000
    samples = np.arange(2 * rate) / rate
    targets = [.4, .9, 1.4]
    signal = np.zeros_like(samples)
    for cue in targets:
        offset = samples - cue
        active = (offset >= 0) & (offset < .08)
        signal[active] = np.sin(2 * np.pi * 440 * offset[active])
    hop = rate // 100
    rms = np.sqrt(np.mean(signal.reshape(-1, hop) ** 2, axis=1))
    active = rms > .3
    detected = np.flatnonzero(active & ~np.r_[False, active[:-1]]) / 100
    assert len(detected) == len(targets) and np.allclose(detected, targets, atol=.01)
    return {"sample_rate": rate, "analysis_window_ms": 10, "cue_seconds": detected.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--expect", required=True, help="Required substring in the GL renderer name")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", action="store_true",
                        help="Write 1080p observed-mode packets as JSON Lines instead of a benchmark report")
    args = parser.parse_args()
    if args.frames < 24 or args.repeat < 1:
        parser.error("Use at least 24 frames and one repetition")
    ctx = moderngl.create_standalone_context(backend="egl", device_index=args.device)
    renderer = ctx.info["GL_RENDERER"]
    if args.expect not in renderer:
        raise SystemExit(f"Wrong renderer: {renderer}")
    if args.stream:
        with args.output.open("x", encoding="utf-8") as output:
            run(ctx, 1920, 1080, "observed", args.frames, packet_sink=output)
        return
    cases = [(w,h,mode) for w,h in ((1280,720),(1920,1080),(2560,1440))
             for mode in ("full","coarse","hybrid","adaptive","observed")]
    rows = []
    for repetition in range(args.repeat):
        # Alternate traversal to reduce order and heat bias.
        for width,height,mode in cases if repetition % 2 == 0 else reversed(cases):
            rows.append({"repeat": repetition + 1, **run(ctx,width,height,mode,args.frames)})
    report = {"measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "renderer": renderer, "gl_version": ctx.info["GL_VERSION"],
              "source_clock_hz": SOURCE_HZ, "audio": audio_check(), "runs": rows}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(renderer, len(rows), "runs", args.output)


if __name__ == "__main__":
    main()
