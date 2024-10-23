import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter


def best_f1(true, pos_logits):
    """Find optimal threshold for F1 score.

    true = [1, 0, 0, 1]
    pos_logits = [0.27292988, 0.27282527, 0.7942509, 0.20574914]
    """
    precision, recall, thresholds = precision_recall_curve(true, pos_logits)
    thresh_scores = []
    for i in range(len(thresholds)):
        if precision[i] + recall[i] == 0:
            continue
        f1 = (2 * (precision[i] * recall[i])) / (precision[i] + recall[i])
        thresh = thresholds[i]
        thresh_scores.append([f1, thresh])
    thresh_scores = sorted(thresh_scores, reverse=True)
    thresh_scores = [i for i in thresh_scores if i[0] > 0]
    return thresh_scores[0][-1]


def get_metrics(true, pred):
    """Get relevant metrics given true labels and logits."""
    metrics = {}
    metrics["acc"] = accuracy_score(true, pred)
    metrics["f1"] = f1_score(true, pred, zero_division=0)
    metrics["rec"] = recall_score(true, pred, zero_division=0)
    metrics["prec"] = precision_score(true, pred, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(true, pred)
    metrics["fpr"] = -1
    metrics["fnr"] = -1
    if sum(true + pred) != 0:
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        if fp + tn != 0:
            metrics["fpr"] = fp / (fp + tn)
        if fn + tp != 0:
            metrics["fnr"] = fn / (fn + tp)
    return metrics


def get_metrics_logits(true, logits):
    """Call get_metrics with logits."""
    loss = F.cross_entropy(logits, true).detach().cpu().item()
    if torch.is_tensor(true):
        true_oh = torch.nn.functional.one_hot(true).detach().cpu().numpy()
        true = true.detach().cpu().numpy()
    if torch.is_tensor(logits):
        sm_logits = torch.nn.functional.softmax(logits, dim=1)
        pos_logits = sm_logits[:, 1].detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
    f1_threshold = best_f1(true, pos_logits)
    pred = [1 if i > f1_threshold else 0 for i in pos_logits]
    try:
        roc_auc = roc_auc_score(true, logits[:, 1])
    except:
        roc_auc = -1
    try:
        pr_auc = average_precision_score(true_oh, logits)
    except:
        pr_auc = -1
    ret = get_metrics(true, pred)
    ret["roc_auc"] = roc_auc
    ret["pr_auc"] = pr_auc
    ret["pr_auc_pos"] = average_precision_score(true, logits[:, 1])
    ret["loss"] = loss
    return ret


def met_dict_to_str(md, prefix="", verbose=1):
    """Convert metric dictionary to string for printing."""
    ret_str = prefix
    for k, v in md.items():
        if k == "loss":
            ret_str += k + ": " + "%.5f" % v + " | "
        else:
            ret_str += k + ": " + "%.3f" % v + " | "
    if verbose > 0:
        print("\x1b[40m\x1b[37m" + ret_str[:-1] + "\x1b[0m")
    return ret_str


def met_dict_to_writer(md, step, writer, prefix):
    """Given a dict of eval metrics, write to given Tensorboard writer."""
    for k, v in md.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


def print_seperator(strings: list, max_len: int):
    """Print text inside a one-line string with "=" seperation to a max length.

    Args:
        strings (list): List of strings.
        max_len (int): Max length.
    """
    midpoints = int(max_len / len(strings))
    strings = [str(i) for i in strings]
    final_str = ""
    cutoff = max_len + (9 * len(strings))
    for s in strings:
        if "\x1b" in s:
            cutoff += 9
        len_s = len(s.replace("\x1b[32m", "").replace("\x1b[39m", ""))
        final_str += "\x1b[40m"
        final_str += "=" * (int((midpoints / 2) - int(len_s / 2)) - 1)
        final_str += f" {s} "
        final_str += "=" * (int((midpoints / 2) - int(len_s / 2)) - 1)
        final_str += "\x1b[0m"
    print(final_str[:cutoff])


def dict_mean(dict_list):
    """Get mean of values from list of dicts.

    https://stackoverflow.com/questions/29027792
    """
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list if not np.isnan(d[key])) / len(
            [d[key] for d in dict_list if not np.isnan(d[key])]
        )
    return mean_dict


class LogWriter:
    """Writer class for logging PyTorch model performance."""

    def __init__(
        self,
        model,
        path: str,
        max_patience: int = 100,
        log_every: int = 10,
        val_every: int = 50,
    ):
        """Init writer.

        Args:
            model: Pytorch model.
            path (str): Path to save log files.
        """
        self._model = model
        self._best_val_loss = 100
        self._patience = 0
        self._max_patience = max_patience
        self._epoch = 0
        self._step = 0
        self._path = Path(path)
        self._writer = SummaryWriter(path)
        self._log_every = log_every
        self._val_every = val_every
        self.save_attrs = ["_best_val_loss", "_patience", "_epoch", "_step"]

    def log(self, train_mets, val_mets):
        """Log information."""
        if self._step % self._log_every != 0:
            self.step()
            return

        if not self.log_val():
            met_dict_to_str(train_mets, "TR = ")
            met_dict_to_writer(train_mets, self._step, self._writer, "TRN")
            self.step()
            return

        val_loss = val_mets["loss"]
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            with open(self._path / "best.model", "wb") as f:
                torch.save(self._model.state_dict(), f)
            best_model_string = "Best model saved: %.3f" % val_loss
            best_model_string = f"\x1b[32m{best_model_string}\x1b[39m"
            self._patience = 0
        else:
            self._patience += 1
            best_model_string = "No improvement."
        print_seperator(
            [
                f"Patience: {self._patience:03d}",
                f"Epoch: {self._epoch:03d}",
                f"Step: {self._step:03d}",
                best_model_string,
            ],
            131,
        )
        met_dict_to_str(train_mets, "TR = ")
        met_dict_to_writer(train_mets, self._step, self._writer, "TRN")
        met_dict_to_str(val_mets, "VA = ")
        met_dict_to_writer(val_mets, self._step, self._writer, "VAL")
        self.step()

    def test(self, test_mets):
        """Helper function to write test mets."""
        print_seperator(["\x1b[36mTest Set\x1b[39m"], 135)
        met_dict_to_str(test_mets, "TS = ")

    def log_val(self):
        """Check whether should validate or not."""
        if self._step % self._val_every == 0:
            return True
        return False

    def step(self):
        """Increment step."""
        self._step += 1

    def epoch(self):
        """Increment epoch."""
        self._epoch += 1

    def stop(self):
        """Check if should stop training."""
        return self._patience > self._max_patience

    def load_best_model(self):
        """Load best model."""
        torch.cuda.empty_cache()
        self._model.load_state_dict(torch.load(self._path / "best.model"))

    def save_logger(self):
        """Save class attributes."""
        with open(self._path / "log.pkl", "wb") as f:
            f.write(pkl.dumps(dict([(i, getattr(self, i)) for i in self.save_attrs])))
        with open(self._path / "current.model", "wb") as f:
            torch.save(self._model.state_dict(), f)

    def load_logger(self):
        """Load class attributes."""
        with open(self._path / "log.pkl", "rb") as f:
            attrs = pkl.load(f)
            for k, v in attrs.items():
                setattr(self, k, v)
        torch.cuda.empty_cache()
        self._model.load_state_dict(torch.load(self._path / "current.model"))




import os
import pickle as pkl


# def get_dep_add_lines(filepath_before, filepath_after, added_lines):
#     """Get lines that are dependent on added lines.

#     Example:
#     df = svdd.bigvul()
#     filepath_before = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/177775.c"
#     filepath_after = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/after/177775.c"
#     added_lines = df[df.id==177775].added.item()

#     """
#     before_graph = feature_extraction(filepath_before)[0]
#     after_graph = feature_extraction(filepath_after)[0]

#     # Get nodes in graph corresponding to added lines
#     added_after_lines = after_graph[after_graph.id.isin(added_lines)]

#     # Get lines dependent on added lines in added graph
#     dep_add_lines = added_after_lines.data.tolist() + added_after_lines.control.tolist()
#     dep_add_lines = set([i for j in dep_add_lines for i in j])

#     # Filter by lines in before graph
#     before_lines = set(before_graph.id.tolist())
#     dep_add_lines = sorted([i for i in dep_add_lines if i in before_lines])

#     return dep_add_lines


# def helper(row):
#     """Run get_dep_add_lines from dict.

#     Example:
#     df = svdd.bigvul()
#     added = df[df.id==177775].added.item()
#     removed = df[df.id==177775].removed.item()
#     helper({"id":177775, "removed": removed, "added": added})
#     """
#     before_path = str(svd.processed_dir() / f"bigvul/before/{row['id']}.c")
#     after_path = str(svd.processed_dir() / f"bigvul/after/{row['id']}.c")
#     try:
#         dep_add_lines = get_dep_add_lines(before_path, after_path, row["added"])
#     except Exception:
#         dep_add_lines = []
#     return [row["id"], {"removed": row["removed"], "depadd": dep_add_lines}]


# def get_dep_add_lines_bigvul(cache=True):
#     """Cache dependent added lines for bigvul."""
#     saved = get_dir(processed_dir() / "bigvul/eval") / "statement_labels.pkl"
#     if os.path.exists(saved) and cache:
#         with open(saved, "rb") as f:
#             return pkl.load(f)
#     df = bigvul()
#     df = df[df.vul == 1]
#     desc = "Getting dependent-added lines: "
#     lines_dict = svd.dfmp(df, helper, ["id", "removed", "added"], ordr=False, desc=desc)
#     lines_dict = dict(lines_dict)
#     with open(saved, "wb") as f:
#         pkl.dump(lines_dict, f)
#     return lines_dict


def eval_statements(sm_logits, labels, thresh=0.5):
    """Evaluate statement-level detection according to IVDetect.

    sm_logits = [
        [0.5747372, 0.4252628],
        [0.53908646, 0.4609135],
        [0.49043426, 0.5095658],
        [0.65794635, 0.34205365],
        [0.3370166, 0.66298336],
        [0.55573744, 0.4442625],
    ]
    labels = [0, 0, 0, 0, 1, 0]
    """
    if sum(labels) == 0:
        preds = [i for i in sm_logits if i[1] > thresh]
        if len(preds) > 0:
            ret = {k: 0 for k in range(1, 11)}
        else:
            ret = {k: 1 for k in range(1, 11)}
    else:
        zipped = list(zip(sm_logits, labels))
        zipped = sorted(zipped, key=lambda x: x[0][1], reverse=True)
        ret = {}
        for i in range(1, 11):
            if 1 in [i[1] for i in zipped[:i]]:
                ret[i] = 1
            else:
                ret[i] = 0
    return ret


def eval_statements_inter(stmt_pred_list, thresh=0.5):
    """Intermediate calculation."""
    total = len(stmt_pred_list)
    ret = {k: 0 for k in range(1, 11)}
    for item in stmt_pred_list:
        eval_stmt = eval_statements(item[0], item[1], thresh)
        for i in range(1, 11):
            ret[i] += eval_stmt[i]
    ret = {k: v / total for k, v in ret.items()}
    return ret


def eval_statements_list(stmt_pred_list, thresh=0.5, vo=False):
    """Apply eval statements to whole list of preds.

    item1 = [[[0.1, 0.9], [0.6, 0.4], [0.4, 0.5]], [0, 1, 1]]
    item2 = [[[0.9, 0.1], [0.6, 0.4]], [0, 0]]
    item3 = [[[0.1, 0.9], [0.6, 0.4]], [1, 1]]
    stmt_pred_list = [item1, item2, item3]
    """
    vo_list = [i for i in stmt_pred_list if sum(i[1]) > 0]
    vulonly = eval_statements_inter(vo_list, thresh)
    if vo:
        return vulonly
    nvo_list = [i for i in stmt_pred_list if sum(i[1]) == 0]
    nonvulnonly = eval_statements_inter(nvo_list, thresh)
    ret = {}
    for i in range(1, 11):
        ret[i] = vulonly[i] * nonvulnonly[i]
    return ret


import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    """Symmetric Cross Entropy Loss.

    https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
    """

    def __init__(self, alpha=1, beta=1, num_classes=2):
        """init."""
        super(SCELoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        """Forward."""
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels, self.num_classes)
            .float()
            .to(self.device)
        )
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


