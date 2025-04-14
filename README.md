# Source Camera Model Identification

This repository contains implementations of various **SCMI (Source Camera Model Identification)** methods. The process involves two major phases:

1. **Patching of Images**
2. **Model Training**

## 📁 Folder Structure

Ensure your dataset is pre-split into `train`, `test`, and `val` folders in the following structure:

```
data/
├── train/
├── test/
└── val/
```

---

## 🔧 Available SCMI Methods

The following SCMI methods are currently supported:

1. **Liu et al.** *(default)*
2. **Bennabhaktula et al.**
3. **Rana et al.**
4. **Syachanran et al.**
5. **Huan et al.**
6. **Rafi et al.**

---

## 🧩 Step 1: Image Patching

All patching strategies are implemented in the `patch/` folder, following the parameters defined in their respective papers. You can customize these by passing arguments while running the script.

### ➤ Script: `patching.py`

**Arguments:**

| Argument      | Type | Default      | Description                                      |
|---------------|------|--------------|--------------------------------------------------|
| `--src`       | str  | `./data`     | Source data directory                            |
| `--dest`      | str  | `./patches`  | Destination to save patches                      |
| `--method`    | str  | `liu`        | Patching method (`liu`, `bennabhaktula`, `rana`, `sychanran`, `huan`, `rafi`) |

### ▶️ Example

To apply patching using the method by **Rana et al.** and save patches to a folder named `patchesRana`:

```bash
python patching.py --method=rana --src=DATA --dest=patchesRana
```

---

## 🧠 Step 2: Model Training

Once patching is complete, the resulting folder (e.g., `patches/`) should contain `train`, `test`, and `val` directories.

### ➤ Script: `main.py`

**Arguments:**

| Argument        | Type   | Default         | Description                                             |
|------------------|--------|-----------------|---------------------------------------------------------|
| `--src`          | str    | `./patches`      | Directory containing patched data                      |
| `--dest`         | str    | `./Result`       | Directory to save prediction results (CSV)             |
| `--onlyTest`     | bool   | `False`          | If `True`, only testing will be performed              |
| `--method`       | str    | `liu`            | SCMI method to use (same options as above)             |
| `--epochs`       | int    | `1`              | Number of training epochs                              |
| `--batchsize`    | int    | `64`             | Batch size                                              |
| `--lr`           | float  | `0.01`           | Learning rate                                           |
| `--momentum`     | float  | `0.9`            | SGD momentum                                            |
| `--weightDecay`  | float  | `0.00075`        | Weight decay (L2 regularization)                       |
| `--modelname`    | str    | `bestModel.pth`  | Name of the saved model file                           |
| `--log`          | str    | `log.csv`        | File to log training statistics                        |
| `--numWorkers`   | int    | `4`              | Number of workers for data loading                     |
| `--device`       | str    | `cuda:0`         | Device to run training (`cuda:i` or `cpu`)             |

### ▶️ Example

To train using the **Rana et al.** method for 10 epochs:

```bash
python main.py --method=rana --src=patchesRana --epochs=10 --device=cuda:0
```

---

## ✅ Notes

- Make sure your dataset is correctly split and organized before running any script.
- You can modify hyperparameters as needed by passing them via command line.
- Logs and results will be saved in the specified `--log` and `--dest` directories respectively.

---

