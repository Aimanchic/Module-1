"""Save reproducible training logs, weights and plots."""

import json as js
import random as rng
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import minitorch as mt
from project.run_scalar import ScalarTrain


def run(nam, hid=10, nep=500):
    rng.seed(0)
    dat = mt.datasets[nam](50)
    net = ScalarTrain(hid)
    dst = Path("results")
    dst.mkdir(exist_ok=True)
    txt = []
    ls = []

    def log(ep, err, num, arr):
        row = f"Epoch {ep:4d}  loss {err:.6f}  correct {num}/50"
        print(nam, row, flush=True)
        txt.append(row)
        ls[:] = arr

    net.train(dat, 0.5, nep, log)
    ys = [net.run_one(p).data for p in dat.X]
    num = sum(int(y > 0.5) == c for y, c in zip(ys, dat.y))
    err = -sum(mt.operators.log(y if c else 1.0 - y) for y, c in zip(ys, dat.y))
    row = f"Final after update: loss {err:.6f}  correct {num}/50"
    txt.append(row)
    print(nam, row, flush=True)
    (dst / f"{nam.lower()}.txt").write_text("\n".join(txt) + "\n")
    rec = {"dataset": nam, "seed": 0, "hidden": hid, "epochs": nep,
           "rate": 0.5, "correct": num, "loss": err, "losses": ls,
           "points": dat.X, "labels": dat.y,
           "weights": {k: p.value.data for k, p in net.model.named_parameters()}}
    (dst / f"{nam.lower()}.json").write_text(js.dumps(rec, indent=2) + "\n")
    fig = draw(nam, dat, net, ls, num)
    fig.savefig(dst / f"{nam.lower()}.png", dpi=150)
    plt.close(fig)


def draw(nam, dat, net, ls, num):
    xs = np.linspace(0, 1, 61)
    x, y = np.meshgrid(xs, xs)
    z = np.array([[net.run_one((a, b)).data for a in xs] for b in xs])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].contourf(x, y, z, levels=[0, 0.5, 1], colors=["#d7e9f7", "#ffe0c2"])
    ax[0].contour(x, y, z, levels=[0.5], colors="black", linewidths=1)
    for cls, col, mrk in [(0, "#2274a5", "o"), (1, "#c45b16", "x")]:
        pts = [p for p, c in zip(dat.X, dat.y) if c == cls]
        ax[0].scatter([p[0] for p in pts], [p[1] for p in pts],
                      c=col, marker=mrk, label=f"Class {cls}")
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y",
              title=f"{nam}: {num}/{dat.N} training points")
    ax[0].legend(loc="upper right")
    ax[1].plot(range(1, len(ls) + 1), ls, color="#2274a5")
    ax[1].set(xlabel="Epoch", ylabel="Total binary cross-entropy", title="Training loss")
    ax[1].grid(alpha=0.2)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for nam in sys.argv[1:] or ["Simple", "Diag", "Split", "Xor"]:
        run(nam)
