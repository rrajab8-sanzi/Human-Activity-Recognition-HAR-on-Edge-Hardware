"""
HAR 1D-CNN: FP32 vs INT8 (TFLite post-training quantization) on SYNTHETIC IMU windows.
Reproduces the numbers used in the report. Run:  pip install tensorflow numpy ; python har_experiment.py
Everything here is simulated data (no real sensor). Latency here is HOST CPU, not ESP32.
"""
import numpy as np, json, time, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
SEED = 42
rng = np.random.default_rng(SEED); tf.random.set_seed(SEED)
FS, WIN = 100, 128
CLASSES = ["Stationary", "Walking", "Fall"]

def unit(v): return v / np.linalg.norm(v)

def make_subject():
    return dict(tilt=unit(rng.normal(size=3) * np.array([0.3, 0.3, 1.0]) + np.array([0, 0, 1.0])),
                f=rng.uniform(1.4, 2.1), amp=rng.uniform(0.6, 1.4), noise=rng.uniform(0.02, 0.05))

def window(cls, s):
    t = np.arange(WIN) / FS
    g = s["tilt"].copy()
    acc = np.tile(g, (WIN, 1)); gyr = np.zeros((WIN, 3))
    if cls in (1, 2):  # walking motion (also precedes a fall)
        ph = rng.uniform(0, 2 * np.pi, 3); f = s["f"] * rng.uniform(0.9, 1.1); a = s["amp"]
        acc[:, 2] += a * 0.35 * np.sin(2 * np.pi * f * t + ph[0]) + a * 0.1 * np.sin(4 * np.pi * f * t + ph[1])
        acc[:, 0] += a * 0.2 * np.sin(2 * np.pi * f * t + ph[2])
        acc[:, 1] += a * 0.1 * np.sin(np.pi * f * t + ph[1])
        gyr[:, 0] += a * 40 * np.sin(2 * np.pi * f * t + ph[0]); gyr[:, 1] += a * 25 * np.sin(np.pi * f * t + ph[2])
    if cls == 2:  # free-fall, impact, then lying in a new orientation
        i0 = rng.integers(20, 70); ff = rng.integers(25, 45); imp = i0 + ff
        acc[i0:imp] *= rng.uniform(0.1, 0.4)
        peak = rng.uniform(3.0, 6.5)
        n = 4; dirv = unit(rng.normal(size=3) + np.array([0, 0, 1.0]))
        for k in range(n): acc[imp + k] = dirv * peak * np.exp(-k / 1.5) * rng.choice([1, -1]) if k % 2 == 0 else -dirv * peak * 0.4 * np.exp(-k / 1.5)
        newg = unit(rng.normal(size=3) * np.array([1, 1, 0.2]))
        acc[imp + n:] = newg + rng.normal(scale=0.03, size=(WIN - imp - n, 3))
        gyr[i0:imp + n, :] += rng.normal(scale=150, size=(imp + n - i0, 3)) + rng.uniform(100, 300, 3)
    acc += rng.normal(scale=s["noise"], size=acc.shape)
    gyr += rng.normal(scale=1.0, size=gyr.shape) + rng.normal(scale=0.5, size=3)
    return np.concatenate([acc, gyr], 1).astype(np.float32)

def dataset(n_subj, per_class):
    X, y = [], []
    for _ in range(n_subj):
        s = make_subject()
        for c in range(3):
            for _ in range(per_class): X.append(window(c, s)); y.append(c)
    return np.array(X), np.array(y)

Xtr_raw, ytr = dataset(18, 100); Xte_raw, yte = dataset(6, 100)

def run(acc_range_g, tag):
    clip = lambda X: np.concatenate([np.clip(X[..., :3], -acc_range_g, acc_range_g), np.clip(X[..., 3:], -250, 250)], -1)
    Xtr, Xte = clip(Xtr_raw), clip(Xte_raw)
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    m = tf.keras.Sequential([
        tf.keras.layers.Input((WIN, 6)),
        tf.keras.layers.Conv1D(16, 5, activation="relu"), tf.keras.layers.MaxPool1D(2),
        tf.keras.layers.Conv1D(32, 5, activation="relu"), tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(3, activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(Xtr, ytr, epochs=25, batch_size=64, validation_split=0.1, verbose=0)
    # FP32 TFLite
    fp = tf.lite.TFLiteConverter.from_keras_model(m).convert()
    # INT8 TFLite (full-integer PTQ)
    N_CAL = 200
    cal_idx = rng.choice(len(Xtr), N_CAL, replace=False)
    def rep():
        for i in cal_idx: yield [Xtr[i:i + 1]]
    cv = tf.lite.TFLiteConverter.from_keras_model(m)
    cv.optimizations = [tf.lite.Optimize.DEFAULT]; cv.representative_dataset = rep
    cv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    cv.inference_input_type = tf.int8; cv.inference_output_type = tf.int8
    q = cv.convert()

    def evaluate(blob, quant):
        it = tf.lite.Interpreter(model_content=blob, num_threads=1); it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        preds, times = [], []
        for x in Xte:
            xi = x[None]
            if quant:
                sc, zp = inp["quantization"]; xi = np.clip(np.round(xi / sc + zp), -128, 127).astype(np.int8)
            it.set_tensor(inp["index"], xi)
            t0 = time.perf_counter(); it.invoke(); times.append((time.perf_counter() - t0) * 1e3)
            preds.append(int(np.argmax(it.get_tensor(out["index"]))))
        preds = np.array(preds)
        rec = [float((preds[yte == c] == c).mean()) for c in range(3)]
        prec = [float((yte[preds == c] == c).mean()) if (preds == c).any() else 0.0 for c in range(3)]
        f1 = [2 * p * r / (p + r) if p + r else 0 for p, r in zip(prec, rec)]
        # arena estimate: peak sum of two simultaneously-live activation tensors
        sizes = [int(np.prod(d["shape"])) * (1 if quant else 4) for d in it.get_tensor_details() if len(d["shape"]) >= 2 and d["shape"][0] == 1]
        return dict(size_kb=len(blob) / 1024, acc=float((preds == yte).mean() * 100), recall=rec, f1=f1,
                    macro_f1=float(np.mean(f1)), host_ms=float(np.median(times)), act_tensor_bytes=sizes,
                    input_scale=(float(inp["quantization"][0]), int(inp["quantization"][1])) if quant else None)
    r = dict(tag=tag, params=int(m.count_params()), fp32=evaluate(fp, False), int8=evaluate(q, True))
    return r

res = [run(2.0, "+-2g"), run(8.0, "+-8g")]
macs = 124 * 16 * (5 * 6) + 58 * 32 * (5 * 16) + 32 * 3
print(json.dumps(dict(macs=macs, results=res, n_train=len(ytr), n_test=len(yte)), indent=1))
