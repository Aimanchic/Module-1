"""Train a network of scalar operations with gradient descent."""

import random as rng

import minitorch as mt


class Network(mt.Module):
    def __init__(obj, hid):
        super().__init__()
        obj.a = Linear(2, hid)
        obj.b = Linear(hid, hid)
        obj.c = Linear(hid, 1)

    def forward(obj, x):
        x = [v.relu() for v in obj.a(x)]
        x = [v.relu() for v in obj.b(x)]
        return obj.c(x)[0].sigmoid()


class Linear(mt.Module):
    def __init__(obj, ni, no):
        super().__init__()
        obj.w = []
        obj.b = []
        for i in range(ni):
            row = []
            for j in range(no):
                val = mt.Scalar(2 * (rng.random() - 0.5))
                row.append(obj.add_parameter(f"w_{i}_{j}", val))
            obj.w.append(row)
        for j in range(no):
            val = mt.Scalar(2 * (rng.random() - 0.5))
            obj.b.append(obj.add_parameter(f"b_{j}", val))

    def forward(obj, xs):
        ys = [b.value for b in obj.b]
        for i, x in enumerate(xs):
            for j in range(len(ys)):
                ys[j] = ys[j] + x * obj.w[i][j].value
        return ys


def default_log_fn(ep, err, num, ls):
    print(f"Epoch {ep:4d}  loss {err:.6f}  correct {num}")


class ScalarTrain:
    def __init__(obj, hid):
        obj.hid = hid
        obj.model = Network(hid)

    def run_one(obj, x):
        return obj.model((mt.Scalar(x[0]), mt.Scalar(x[1])))

    def train(obj, dat, lr, nep=500, log=default_log_fn):
        obj.model = Network(obj.hid)
        opt = mt.SGD(obj.model.parameters(), lr)
        ls = []
        for ep in range(1, nep + 1):
            err = 0.0
            num = 0
            opt.zero_grad()
            for (x, y), cls in zip(dat.X, dat.y):
                out = obj.model((mt.Scalar(x), mt.Scalar(y)))
                num += int((out.data > 0.5) == cls)
                prb = out if cls == 1 else 1.0 - out
                val = -prb.log()
                (val / dat.N).backward()
                err += val.data
            ls.append(err)
            opt.step()
            if ep % 10 == 0 or ep == nep:
                log(ep, err, num, ls)


if __name__ == "__main__":
    rng.seed(0)
    dat = mt.datasets["Simple"](50)
    ScalarTrain(2).train(dat, 0.5)
