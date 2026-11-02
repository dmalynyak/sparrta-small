from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEAT_DIR = Path("features")
OUT_DIR = Path("results")
SEEDS = [0, 1, 2]
VAL_SIZE = 0.125 # 1/8 of trainval
CS = np.logspace(-3, 2, 11)  # 0.001 ... 100
MAX_ITER = 5000

# RELATIONS = ["left", "right", "front", "back"]
# labels: left-0 right-1 front-2 back-3
TASKS = {"4way": [0, 1, 2, 3], "left_right": [0, 1], "front_back": [2, 3]}

VIT_FEATURES = {"vit_sup": ["cls", "mean"], "vit_dino": ["cls", "mean"]}
RESNET_FEATURES = {"resnet50": ["layer1", "layer2", "layer3", "layer4"]}


def iter_features():
    # yields: 
        # model(str):     vit_sup   vit_dino   resnet50
        # pool(str):      cls,mean  cls,mean   gap
        # layer(int):     1-12      1-12       1-4
        # X(1000xD):      1000x768  1000x768   1000xD (256,512,1024,2048)
        # data(dict):   whole data + metadata from extract_features. 

    for model, poolings in VIT_FEATURES.items():
        data = np.load(FEAT_DIR / f"{model}.npz")
        for pooling in poolings:
            feats = data[pooling]  # 1000 x 12 x 768
            for layer in range(feats.shape[1]):
                yield model, pooling, layer + 1, feats[:, layer, :], data
    for model, layers in RESNET_FEATURES.items():
        data = np.load(FEAT_DIR / f"{model}.npz")
        for layer, name in enumerate(layers):
            yield model, "gap", layer + 1, data[name], data  # 1000 x D


def task_indices(labels, splits, classes):
    # labels (1000, ) 0-3 , splits (1000, ) str , classes (4, ) or (2, )
    in_task = np.isin(labels, classes)  # (1000, ) bool
    tv_idx = np.where(in_task & (splits == "trainval"))[0]
    test_idx = np.where(in_task & (splits == "test"))[0]
    return tv_idx, test_idx  # 4way: (800, )(200, ), binary: (400, )(100, )


def probe(X_train, y_train, X_val, y_val, X_test, y_test):
    scaler = StandardScaler().fit(X_train)  # normalization values based on training dataset
    X_train, X_val, X_test = (scaler.transform(X) for X in (X_train, X_val, X_test))

    best = None
    for C in CS:  # ascending, so ties on val keep the smaller C (stronger regularization)
        probe_head = LogisticRegression(C=C, max_iter=MAX_ITER).fit(X_train, y_train) # 700xD and 700,
        val_acc = probe_head.score(X_val, y_val) # 100xD and 100,  Right/All
        if best is None or val_acc > best[1]:
            best = (C, val_acc, probe_head)
    C, val_acc, probe_head = best
    return dict(C=C, train_acc=probe_head.score(X_train, y_train), val_acc=val_acc,
                test_acc=probe_head.score(X_test, y_test))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for model, pooling, layer, X, data in iter_features():
        # model(str):     vit_sup   vit_dino   resnet50
        # pool(str):      cls,mean  cls,mean   gap
        # layer(int):     1-12      1-12       1-4
        # X(1000xD):      1000x768  1000x768   1000xD (256,512,1024,2048)
        # data(dict):   whole data + metadata from extract_features. 

        labels, splits = data["labels"], data["splits"] # (1000, ) 0-3 (1000, ) str
        for task, classes in TASKS.items():
            tv_idx, test_idx = task_indices(labels, splits, classes)
            for seed in SEEDS:
                train_idx, val_idx = train_test_split(tv_idx, test_size=VAL_SIZE,
                                                      stratify=labels[tv_idx], random_state=seed)

                # 4way:   train_idx, val_idx, test_idx (700,)(100,)(200,) all int(0-999)
                # binary: train_idx, val_idx, test_idx (350,)(50,)(100,)  all int(0-999)
                res = probe(X[train_idx], labels[train_idx], X[val_idx], labels[val_idx],
                            X[test_idx], labels[test_idx])
                rows.append(dict(model=model, pooling=pooling, layer=layer, task=task, seed=seed, **res))
                print(f"{model}, {pooling} layer: {layer}, {task}, seed: {seed}")


    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    

if __name__ == "__main__":
    main()
