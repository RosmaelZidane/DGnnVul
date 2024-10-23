from data_preparation import *
from graphextractor import *
import json
from glob import glob
from pathlib import Path
import pandas as pd

#################### Codebert_________________________

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from important import *

class BigVulDatasetNLP:
    """Override getitem for codebert."""

    def __init__(self, partition="train", random_labels=False):
        """Init."""
        self.df = bigvul()
        self.df = self.df[self.df.label == partition]
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0] .sample(len(vul), random_state=0) # brinh this back when you have over 1000 sample functions ---------->>
            self.df = pd.concat([vul, nonvul])
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        text = [tokenizer.sep_token + " " + ct for ct in self.df.before.tolist()]
        tokenized = tokenizer(text, **tk_args)
        self.labels = self.df.vul.tolist()
        if random_labels:
            self.labels = torch.randint(0, 2, (len(self.df),)).tolist()
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Override getitem."""
        return self.ids[idx], self.att_mask[idx], self.labels[idx]


class BigVulDatasetNLPLine:
    """Override getitem for codebert."""

    def __init__(self, partition="train"):
        """Init."""
        linedict = get_dep_add_lines_bigvul()
        df = bigvul()
        df = df[df.label == partition]
        df = df[df.vul == 1].copy()
        df = df.sample(min(1000, len(df))) # it does not work, bring back the code #---------------->>

        texts = []
        self.labels = []

        for row in df.itertuples():
            line_info = linedict[row.id]
            vuln_lines = set(list(line_info["removed"]) + line_info["depadd"])
            for idx, line in enumerate(row.before.splitlines(), start=1):
                line = line.strip()
                if len(line) < 5:
                    continue
                if line[:2] == "//":
                    continue
                texts.append(line.strip())
                self.labels.append(1 if idx in vuln_lines else 0)

        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        text = [tokenizer.sep_token + " " + ct for ct in texts]
        tokenized = tokenizer(text, **tk_args)
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """Get length of dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Override getitem."""
        return self.ids[idx], self.att_mask[idx], self.labels[idx]



class BigVulDatasetNLPDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(self, DataClass, batch_size: int = 32, sample: int = -1):
        """Init class from bigvul dataset."""
        super().__init__()
        self.train = DataClass(partition="train")
        self.val = DataClass(partition="val")
        self.test = DataClass(partition="test")
        self.batch_size = batch_size

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return val dataloader."""
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test, batch_size=self.batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LitCodebert(pl.LightningModule):
    """Codebert."""

    def __init__(self, lr: float = 1e-3):
        """Initilisation."""
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.fc1 = torch.nn.Linear(768, 256)
        self.fc2 = torch.nn.Linear(256, 2)
        self.accuracy = torchmetrics.Accuracy(task = "binary", num_classes = 2 ) 
        self.auroc = torchmetrics.AUROC(task = "binary", num_classes = 2) 
        from torchmetrics import MatthewsCorrCoef
        self.mcc = MatthewsCorrCoef(task = "binary", num_classes = 2)

    def forward(self, ids, mask):
        """Forward pass."""
        with torch.no_grad():
            bert_out = self.bert(ids, attention_mask=mask)
        fc1_out = self.fc1(bert_out["pooler_output"])
        fc2_out = self.fc2(fc1_out)
        return fc2_out
    
    
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        #--------------------------------------------------------------
        labels = labels.type(torch.LongTensor) # let's see
        labels , logits = labels.to(device), logits.to(device)
        #------------------------------
        loss = F.cross_entropy(logits, labels)

        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        self.log("train_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        
        #--------------------------------------
        labels = labels.type(torch.LongTensor) # let's see
        labels , logits = labels.to(device), logits.to(device)
        #------------------------------
        loss = F.cross_entropy(logits, labels)
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.auroc.update(logits[:, 1], labels)
        self.log("val_auroc", self.auroc, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        #--------------------------------------------------------------------
        labels = labels.type(torch.LongTensor) # let's see
        labels , logits = labels.to(device), logits.to(device)
        #---------------------------------------------------------------------
        loss = F.cross_entropy(logits, labels)
        self.auroc.update(logits[:, 1], labels)
        self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

run_id = get_run_id()
savepath = get_dir(processed_dir() / "codebert" / run_id)
from fpdf import FPDF

###===============================================================================================================================
#====================== After the first run, put all this in comment ==============================================

# model = LitCodebert()
# data = BigVulDatasetNLPDataModule(BigVulDatasetNLP, batch_size=64)
# checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
# trainer = pl.Trainer(
#     accelerator= "auto",
#     devices= "auto",
#     max_epochs= 20,
#     default_root_dir=savepath,
#     num_sanity_val_steps=0,
#     callbacks=[checkpoint_callback],
# )

# print(f".......................\n----------------Training CodeBert model----------------\n....................\n")
# trainer.fit(model, data)
# trainer.test(model, data)

# test_results = trainer.test(model, data)
# train_metrics = trainer.callback_metrics

# # Create a PDF document
# pdf = FPDF()

# # Add a page
# pdf.add_page()

# # Set title
# pdf.set_font("Arial", size=12)
# pdf.cell(200, 10, txt="Codebert Model Performance Metrics", ln=True, align='C')

# # Add training metrics
# pdf.set_font("Arial", size=10)
# pdf.cell(200, 10, txt="Training Metrics:", ln=True, align='L')
# for key, value in train_metrics.items():
#     pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align='L')

# pdf.cell(200, 10, txt="", ln=True, align='L')

# # Add testing metrics
# pdf.set_font("Arial", size=10)
# pdf.cell(200, 10, txt="Testing Metrics:", ln=True, align='L')
# for result in test_results:
#     for key, value in result.items():
#         pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align='L')

# # Save the PDF
# pdf.output(outputs_dir() / "codebert_model_metrics.pdf")

# print("PDF of the codebert saved successfully.")

# outputs_dir()
# Comment up tu here
#--------------------------------------------------------------

import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, AutoTokenizer
from tsne_torch import TorchTSNE as TSNE



class CodeBert:
    """CodeBert.

    Example:
    cb = CodeBert()
    sent = ["int myfunciscool(float b) { return 1; }", "int main"]
    ret = cb.encode(sent)
    ret.shape
    >>> torch.Size([2, 768])
    """

    def __init__(self):
        """Initiate model."""
        codebert_base_path = external_dir() / "codebert-base"
        if os.path.exists(codebert_base_path):
            self.tokenizer = AutoTokenizer.from_pretrained(codebert_base_path)
            self.clean_up_tokenization_spaces =  True
            self.model = AutoModel.from_pretrained(codebert_base_path)
        else:
            def cache_dir() -> Path:
                """Get storage cache path."""
                path = storage_dir() / "cache"
                Path(path).mkdir(exist_ok=True, parents=True)
                return path
            cache_dir = get_dir(f"{cache_dir()}/codebert_model")
            print("Loading Codebert...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
        self._dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._dev)

    def encode(self, sents: list):
        """Get CodeBert embeddings from a list of sentences."""
        tokens = [i for i in sents]
        self.clean_up_tokenization_spaces =  True
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer(tokens, **tk_args).to(self._dev)
        with torch.no_grad():
            return self.model(tokens["input_ids"], tokens["attention_mask"])[1]


def plot_embeddings(embeddings, words):
    """Plot embeddings.

    import sastvd.helpers.datasets as svdd
    cb = CodeBert()
    df = svdd.bigvul()
    sent = " ".join(df.sample(5).before.tolist()).split()
    plot_embeddings(cb.encode(sent), sent)
    """
    tsne = TSNE(n_components=2, n_iter=2000, verbose=True)
    Y = tsne.fit_transform(embeddings)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(words, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()
    
    
    
#-============================================================================================

class BigVulDataset:
    """Represent BigVul as graph dataset."""

    def __init__(self, partition="train", vulonly=False, sample=-1, splits="default"):
        """Init class."""
        # Get finished samples
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(processed_dir() / "bigvul/before/*nodes*"))
        ]
        self.df = bigvul(splits=splits)
        self.partition = partition
        self.df = self.df[self.df.label == partition]
        self.df = self.df[self.df.id.isin(self.finished)]

        # Balance training set
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # Correct ratio for test set
        if partition == "test":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]
            nonvul = nonvul.sample(min(len(nonvul), len(vul) * 20), random_state=0) ##----->>>
            self.df = pd.concat([vul, nonvul])

        # Small sample (for debugging):
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)

        # Filter only vulnerable
        if vulonly:
            self.df = self.df[self.df.vul == 1]

        # Filter out samples with no lineNumber from Joern output
        self.df["valid"] = dfmp(
            self.df, BigVulDataset.check_validity, "id", desc="Validate Samples: "
        )
        self.df = self.df[self.df.valid]

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        

    def itempath(_id):
        """Get itempath path from item id."""
        return processed_dir() / f"bigvul/before/{_id}.java"

    def check_validity(_id):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        try:
            with open(str(BigVulDataset.itempath(_id)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:
                            valid = 1
                            break
                if valid == 0:
                    return False
            with open(str(BigVulDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            print(E, str(BigVulDataset.itempath(_id)))
            return False

    def get_vuln_indices(self, _id):
        """Obtain vulnerable lines from sample ID."""
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])

    def stats(self):
        """Print dataset stats."""
        print(self.df.groupby(["label", "vul"]).count()[["id"]])

    def __getitem__(self, idx):
        """Must override."""
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"BigVulDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"



def get_sast_lines(sast_pkl_path):
    """Get sast lines from path to sast dump."""
    ret = dict()
    ret["cppcheck"] = set()
    ret["rats"] = set()
    ret["flawfinder"] = set()

    try:
        with open(sast_pkl_path, "rb") as f:
            sast_data = pkl.load(f)
        for i in sast_data:
            if i["sast"] == "cppcheck":
                if i["severity"] == "error" and i["id"] != "syntaxError":
                    ret["cppcheck"].add(i["line"])
            elif i["sast"] == "flawfinder":
                if "CWE" in i["message"]:
                    ret["flawfinder"].add(i["line"])
            elif i["sast"] == "rats":
                ret["rats"].add(i["line"])
    except Exception as E:
        print(E)
        pass
    return ret



import networkx as nx
import matplotlib.pyplot as plt

def draw_dgl_graph(dgl_graph):
    """Convert DGL graph to NetworkX graph"""
    nx_graph = dgl_graph.to_networkx()
    pos = nx.spring_layout(nx_graph) 
    nx.draw(nx_graph, pos, with_labels=True, 
            node_color='skyblue', node_size=400, 
            edge_color='black', linewidths=1, 
            font_size=8)
 
    plt.show()
#draw_dgl_graph(g)


import networkx as nx
from node2vec import Node2Vec

# Write a function to read the text file on the from the vul-description folder


def read_CVEvuldescription(_idd):
    """This function read domain information from the description folders"""
    pathd = f"{processed_dir()}/bigvul/CVEdescription"
    
    with open(f"{pathd}/{_idd}.txt", 'r') as f:
        content_ = f.read()
    
    return content_
        

class BigVulDatasetLineVD(BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(self, gtype="pdg", feat="all", **kwargs):
        """Init."""
        super(BigVulDatasetLineVD, self).__init__(**kwargs)
        lines = get_dep_add_lines_bigvul()
        lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        self.lines = lines
        self.graph_type = gtype 
        self.feat = feat

    def item(self, _id, codebert=None):
        """Cache item."""
        savedir = get_dir(
            cache_dir() / f"bigvul_linevd_codebert_{self.graph_type}"
        ) / str(_id)
        if os.path.exists(savedir):
            g = load_graphs(str(savedir))[0][0]
            
            if "_CODEBERT" in g.ndata:
                if self.feat == "codebert":
                    for i in ["_GLOVE", "_DOC2VEC", "_RANDFEAT"]:
                        try:
                            g.ndata.pop(i, None)
                        except:
                            print(f"No {i} in nodes feature")
                return g
            
        code, lineno, ei, eo, et = feature_extraction(
            BigVulDataset.itempath(_id), self.graph_type
        )
        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in lineno]
        else:
            vuln = [0 for _ in lineno]
        g = dgl.graph((eo, ei))
        
        if codebert:
            code = [c.replace("\\t", "").replace("\\n", "") for c in code]
            chunked_batches = chunks(code, 128)
            features = [codebert.encode(c).detach().cpu() for c in chunked_batches]
            g.ndata["_CODEBERT"] = th.cat(features)
        g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
        g.ndata["_LINE"] = th.Tensor(lineno).int()
        g.ndata["_VULN"] = th.Tensor(vuln).float()
        g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
        g.edata["_ETYPE"] = th.Tensor(et).long()
        emb_path = cache_dir() / f"codebert_method_level/{_id}.pt"
        g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
        
        # read the description and optain embedding with codebert
        desc_ = read_CVEvuldescription(_idd = _id)
        cb = CodeBert()  
        text_ = [desc_]
        embedd_text = cb.encode(sents = text_).detach().cpu()
        g.ndata['_CVEVuldesc'] = th.Tensor(embedd_text).repeat((g.number_of_nodes(), 1))
        
        
        
        ##%------------------- Remove this part from comment only when you have some thing promissing
        # # Node embeddings step 
        # nx_graph = g.to_networkx() # dimensions=64, walk_length=30, num_walks=200, workers=4)
        # node2vec = Node2Vec(nx_graph, dimensions=768, walk_length=30, num_walks=200, workers=4)
        # model = node2vec.fit(window = 10, min_count = 1, batch_words = 8)
        # embeddings = model.wv
        # node_embeddings = {int(node): embeddings[str(node)] for node in nx_graph.nodes}
        # # print(f"------------------------------------------------------{node_embeddings}")
        # embedding_matrix = torch.tensor([node_embeddings[node.item()] for node in g.nodes()], dtype=torch.float)
        # g.ndata['node_embedding'] = embedding_matrix
        # #g.ndata['node_embedding'] = dgl.utils.Tensor(node_embeddings)
        
        # # edges embedding
        # src, dst = g.edges()
        # src_embeddings = g.ndata['node_embedding'][src]
        # dst_embeddings = g.ndata['node_embedding'][dst]
        # edge_embeddings = (src_embeddings + dst_embeddings) / 2
        # g.edata['edge_embedding'] = edge_embeddings
        
        g = dgl.add_self_loop(g)
        save_graphs(str(savedir), [g])
        return g

    def cache_items(self, codebert):
        """Cache all items."""
        for i in tqdm(self.df.sample(len(self.df)).id.tolist()):
            try:
                self.item(i, codebert)
            except Exception as E:
                print(E)

    def cache_codebert_method_level(self, codebert):
        """Cache method-level embeddings using Codebert.

        ONLY NEEDS TO BE RUN ONCE.
        """
        savedir = get_dir(cache_dir() / "codebert_method_level")
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        done = set(done)
        batches = chunks((range(len(self.df))), 128)
        for idx_batch in tqdm(batches):
            batch_texts = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].before.tolist()
            batch_ids = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].id.tolist()
            if set(batch_ids).issubset(done):
                continue
            texts = ["</s> " + ct for ct in batch_texts]
            embedded = codebert.encode(texts).detach().cpu()
            assert len(batch_texts) == len(batch_ids)
            for i in range(len(batch_texts)):
                th.save(embedded[i], savedir / f"{batch_ids[i]}.pt")

    def __getitem__(self, idx):
        """Override getitem."""
        # print(f"Number of items {self.idx2id[idx]}")
        return self.item(self.idx2id[idx])


class BigVulDatasetLineVDDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(
        self,
        batch_size: int = 32,
        sample: int = -1,
        methodlevel: bool = False,
        nsampling: bool = False,
        nsampling_hops: int = 1,
        gtype: str = "cfgcdg",
        splits: str = "default",
        feat: str = "all",
    ):
        """Init class from bigvul dataset."""
        super().__init__()
        dataargs = {"sample": sample, "gtype": gtype, "splits": splits, "feat": feat}
        self.train = BigVulDatasetLineVD(partition="train", **dataargs)
        self.val = BigVulDatasetLineVD(partition="val", **dataargs)
        self.test = BigVulDatasetLineVD(partition="test", **dataargs)
        codebert = CodeBert()
        self.train.cache_codebert_method_level(codebert)
        self.val.cache_codebert_method_level(codebert)
        self.test.cache_codebert_method_level(codebert)
        self.train.cache_items(codebert)
        self.val.cache_items(codebert)
        self.test.cache_items(codebert)
        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops
                
     
    def node_dl(self, g, shuffle=False):
        """Return node dataloader."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nsampling_hops)
        return dgl.dataloading.DataLoader(
            g,
            g.nodes(),
            sampler,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=10,
        ) 

    def train_dataloader(self):
        """Return train dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.train, batch_size=len(self.train))))
            return self.node_dl(g, shuffle=True)
        return GraphDataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self):
        """Return val dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.val, batch_size=len(self.val), num_workers=10)))
            return self.node_dl(g)
        return GraphDataLoader(self.val, batch_size=self.batch_size, num_workers=10)

    def val_graph_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.val, batch_size=32, num_workers=10)

    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.test, batch_size=32, num_workers=10)

#------------------------------------------

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
#---------------------------------------------



import pytorch_lightning as pl
import torch as th
import torch.nn.functional as F
import torchmetrics
import dgl
from dgl.nn import GATConv, GraphConv
from torch.optim import AdamW


##### ------------ Good cell for now ------------------------------------

class LitGNN(pl.LightningModule):
    """Main Trainer."""

    def __init__(
        self,
        hfeat: int = 512,
        embtype: str = "codebert",
        embfeat: int = -1,  # Keep for legacy purposes
        num_heads: int = 4,
        lr: float = 1e-3,
        hdropout: float = 0.2,
        mlpdropout: float = 0.2,
        gatdropout: float = 0.2,
        methodlevel: bool = False,
        nsampling: bool = False,
        model: str = "gat2layer",
        loss: str = "ce", # "sce", # 
        multitask: str = "linemethod",
        stmtweight: int = 5,
        gnntype: str = "gat",
        random: bool = False,
        scea: float = 0.7,
    ):
        """Initialization."""
        super().__init__()
        self.lr = lr
        self.random = random
        self.save_hyperparameters()

        self.test_step_outputs = []

        # Set params based on embedding type
        if self.hparams.embtype == "codebert":
            self.hparams.embfeat = 768
            self.EMBED = "_CODEBERT"

        # Loss
        if self.hparams.loss == "sce":
            self.loss = SCELoss(self.hparams.scea, 1 - self.hparams.scea)
            self.loss_f = th.nn.CrossEntropyLoss()
        else:
            self.loss = th.nn.CrossEntropyLoss(
                weight=th.Tensor([1, self.hparams.stmtweight]) #.cuda() ---------------??????
            )
            self.loss_f = th.nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.auroc = torchmetrics.AUROC(task="binary", num_classes=2)
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary", num_classes=2)

        # GraphConv Type
        hfeat = self.hparams.hfeat
        gatdrop = self.hparams.gatdropout
        numheads = self.hparams.num_heads
        embfeat = self.hparams.embfeat
        gnn_args = {"out_feats": hfeat}
        if self.hparams.gnntype == "gat":
            gnn = GATConv
            gat_args = {"num_heads": numheads, "feat_drop": gatdrop}
            gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}
            gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads}
        elif self.hparams.gnntype == "gcn":
            gnn = GraphConv
            gnn1_args = {"in_feats": embfeat, **gnn_args}
            gnn2_args = {"in_feats": hfeat, **gnn_args}

        # model: gat2layer
        if "gat" in self.hparams.model:
            self.gat = gnn(**gnn1_args)
            self.gat2 = gnn(**gnn2_args)
            fcin = hfeat * numheads if self.hparams.gnntype == "gat" else hfeat
            self.fc = th.nn.Linear(fcin, self.hparams.hfeat)
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: mlp-only
        if "mlponly" in self.hparams.model:
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: contains femb
        if "+femb" in self.hparams.model:
            self.fc_femb = th.nn.Linear(embfeat * 2, self.hparams.hfeat)

        # Transform codebert embedding
        self.codebertfc = th.nn.Linear(768, self.hparams.hfeat)

        # Hidden Layers
        self.fch = []
        for _ in range(8):
            self.fch.append(th.nn.Linear(self.hparams.hfeat, self.hparams.hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(self.hparams.hdropout)
        self.fc2 = th.nn.Linear(self.hparams.hfeat, 2)

    def forward(self, g, test=False, e_weights=[], feat_override=""):
        """Forward pass."""
        if self.hparams.nsampling and not test:
            hdst = g[2][-1].dstdata[self.EMBED]
            h_func = g[2][-1].dstdata["_FUNC_EMB"]
            g2 = g[2][1]
            g = g[2][0]
            if "gat2layer" in self.hparams.model:
                h = g.srcdata[self.EMBED]
            elif "gat1layer" in self.hparams.model:
                h = g2.srcdata[self.EMBED]
        else:
            g2 = g
            h = g.ndata[self.EMBED]
            if len(feat_override) > 0:
                h = g.ndata[feat_override]
            h_func = g.ndata["_FUNC_EMB"]
            hdst = h

        if self.random:
            return th.rand((h.shape[0], 2)).to(self.device), th.rand(
                h_func.shape[0], 2
            ).to(self.device)

        # model: contains femb
        if "+femb" in self.hparams.model:
            h = th.cat([h, h_func], dim=1)
            h = F.elu(self.fc_femb(h))

        # Transform h_func if wrong size
        if self.hparams.embfeat != 768:
            h_func = self.codebertfc(h_func)

        # model: gat2layer
        if "gat" in self.hparams.model:
            if "gat2layer" in self.hparams.model:
                h = self.gat(g, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
                h = self.gat2(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            elif "gat1layer" in self.hparams.model:
                h = self.gat(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            h = self.mlpdropout(F.elu(self.fc(h)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Edge masking (for GNNExplainer)
        if test and len(e_weights) > 0:
            g.ndata["h"] = h
            g.edata["ew"] = e_weights
            g.update_all(
                dgl.function.u_mul_e("h", "ew", "m"), dgl.function.mean("m", "h")
            )
            h = g.ndata["h"]

        # model: mlp-only
        if "mlponly" in self.hparams.model:
            h = self.mlpdropout(F.elu(self.fconly(hdst)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Hidden layers
        for idx, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h)))
            h_func = self.hdropout(F.elu(hlayer(h_func)))
        h = self.fc2(h)
        h_func = self.fc2(
            h_func
        )  # Share weights between method-level and statement-level tasks

        if self.hparams.methodlevel:
            g.ndata["h"] = h
            return dgl.mean_nodes(g, "h"), None
        else:
            return h, h_func  # Return two values for multitask training

    def shared_step(self, batch, test=False):
        """Shared step."""
        logits = self(batch, test)
        if self.hparams.methodlevel:
            if self.hparams.nsampling:
                raise ValueError("Cannot train on method level with nsampling.")
            labels = dgl.max_nodes(batch, "_VULN").long()
            labels_func = None
        else:
            if self.hparams.nsampling and not test:
                labels = batch[2][-1].dstdata["_VULN"].long()
                labels_func = batch[2][-1].dstdata["_FVULN"].long()
            else:
                labels = batch.ndata["_VULN"].long()
                labels_func = batch.ndata["_FVULN"].long()
        return logits, labels, labels_func
# modification start here
    def training_step(self, batch, batch_idx):
        """Training step."""
        logits, labels, labels_func = self.shared_step(batch)
        loss1 = self.loss(logits[0], labels)
        
        logits1 = logits[0]
        
        if not self.hparams.methodlevel:
            loss2 = self.loss_f(logits[1], labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
        else:
            loss = loss1
            acc_func = self.accuracy(logits, labels_func)
            self.log("train_acc_func", acc_func, prog_bar=True, logger=True)
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_loss_func", loss2, on_epoch=True, prog_bar=True)
        self.log("train_auroc", self.auroc(preds, labels), prog_bar=True)
        self.log("train_acc", self.accuracy(preds, labels), prog_bar=True)
        self.log("train_mcc", self.mcc(preds, labels), prog_bar=True)
        
        if not self.hparams.methodlevel:
            self.log("train_acc_func", self.accuracy(preds_func, labels_func), prog_bar=True)
            self.log("train_auroc_func", self.auroc(preds_func, labels_func), prog_bar=True)
            self.log("train_mcc_func", self.mcc(preds_func, labels_func), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        logits, labels, labels_func = self.shared_step(batch)
        logits1 = logits[0]
        loss1 = self.loss(logits1, labels)
        if not self.hparams.methodlevel:
            logits2 = logits[1]
            loss2 = self.loss_f(logits2, labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
        else:
            loss = loss1
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auroc", self.auroc(preds, labels), prog_bar=True)
        self.log("val_acc", self.accuracy(preds, labels), prog_bar=True)
        self.log("val_mcc", self.mcc(preds, labels), prog_bar=True)

        if not self.hparams.methodlevel:
            self.log("val_acc_func", self.accuracy(preds_func, labels_func), prog_bar=True)
            self.log("val_auroc_func", self.auroc(preds_func, labels_func), prog_bar=True)
            self.log("val_mcc_func", self.mcc(preds_func, labels_func), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        logits, labels, labels_func = self.shared_step(batch, test=True)
        logits1 = logits[0]
        loss1 = self.loss(logits1, labels)
        if not self.hparams.methodlevel:
            logits2 = logits[1]
            loss2 = self.loss_f(logits2, labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
        else:
            loss = loss1
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None

        metrics = {
            "test_loss": loss,
            "test_acc": self.accuracy(preds, labels),
            "test_auroc": self.auroc(preds, labels),
            "test_mcc": self.mcc(preds, labels)
        }
        
        if not self.hparams.methodlevel:
            metrics["test_acc_func"] = self.accuracy(preds_func, labels_func)
            metrics["test_auroc_func"] = self.auroc(preds_func, labels_func)
            metrics["test_mcc_func"] = self.mcc(preds_func, labels_func)

        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        """Test epoch end."""
        avg_metrics = {
            key: th.mean(th.stack([x[key] for x in self.test_step_outputs]))
            for key in self.test_step_outputs[0].keys()
        }
        print(f"what is insight self.test_step_outputs {self.test_step_outputs}")
        self.test_step_outputs.clear()
        self.log_dict(avg_metrics)
        return
        
    def configure_optimizers(self):
        """Configure optimizers."""
        return AdamW(self.parameters(), lr=self.lr)


#  train the classifier

print(f"----------------------->>> Training the model")

model = LitGNN( 
               hfeat= 512,
               embtype= "codebert",
               methodlevel=False,
               nsampling=True,
               model= "gat2layer",
               loss="ce",
               hdropout=0.3,
               gatdropout=0.2,
               num_heads=4,
               multitask="linemethod", 
               stmtweight=1,
               gnntype="gat",
               scea=0.5,
               lr=1e-4,
               )

  # Load data
samplesz = -1
data = BigVulDatasetLineVDDataModule(
    batch_size=64,
    sample=samplesz,
    methodlevel=False,
    nsampling=True,
    nsampling_hops=2,
    gtype= "pdg+raw",
    splits="default",
    )

max_epochs = 5 #110
# # Train model

checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
metrics = ["train_loss", "val_loss", "val_auroc"]
trainer = pl.Trainer(
    accelerator= "auto",
    devices= "auto",
    default_root_dir=savepath,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback], 
    max_epochs=max_epochs,
    )
trainer.fit(model, data)
# trainer.test(model, data)


# g1 = data.test_dataloader().dataset[3] 
# print(g1)

# The code to test the model or make a prediction on a single graph
# from sklearn.metrics import matthews_corrcoef, ndcg_score, f1_score, precision_score
# from sklearn.metrics import recall_score, accuracy_score, roc_auc_score

# def calculate_metrics(model, data):
#     """
#     Calculate ranking metrics: MRR, N@5, MFR,
#     and classification metrics: F1-Score, Precision.
#     """
#     def mean_reciprocal_rank(y_true, y_scores):
#         order = np.argsort(y_scores, axis=1)[:, ::-1]
#         rank = np.argwhere(order == y_true[:, None])[:, 1] + 1
#         return np.mean(1.0 / rank)

#     def precision_at_n(y_true, y_scores, n=5):
#         order = np.argsort(y_scores, axis=1)[:, ::-1]
#         top_n = order[:, :n]
#         return np.mean(np.any(top_n == y_true[:, None], axis=1))

#     def mean_first_rank(y_true, y_scores):
#         order = np.argsort(y_scores, axis=1)[:, ::-1]
#         rank = np.argwhere(order == y_true[:, None])[:, 1] + 1
#         return np.mean(rank)

#     # Extract function-level predictions and true labels
#     all_preds_ = []
#     all_labels_ = []
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()
#     #for batch in data.test_dataloader():
#     batch = data.test_dataloader().dataset[0]
#     path = "/home/New_move/DomainGraph/storage/cache/bigvul_linevd_codebert_pdg+raw/4620"
#     # g = load_graphs(path)[0][0]
#     # batch = g
#     print(f"batch {batch}")
#     with torch.no_grad():
#         logits, labels, labels_func = model.shared_step(batch.to(device), test=True)
#         if labels is not None:
#             preds_ = torch.softmax(logits[0], dim=1).cpu().numpy()
#             labels_f = labels.cpu().numpy()
#             all_preds_.extend(preds_)
#             all_labels_.extend(labels_f)

#     all_preds_ = np.array(all_preds_)
#     all_labels_ = np.array(all_labels_)

#     # Compute ranking metrics
#     MRR = mean_reciprocal_rank(all_labels_, all_preds_)
#     N5 = precision_at_n(all_labels_, all_preds_, n=5)
#     MFR = mean_first_rank(all_labels_, all_preds_)

#     predicted_classes = np.argmax(all_preds_, axis=1)
#     f1_c = f1_score(all_labels_, predicted_classes, average="macro")
#     precision = precision_score(all_labels_, predicted_classes, average="macro")
#     accuracy = accuracy_score(all_labels_, predicted_classes, normalize= True )
#     recall = recall_score(all_labels_, predicted_classes,average = "macro")
#     roc_ = roc_auc_score(all_labels_, predicted_classes, average= "macro")
#     mcc_ = matthews_corrcoef(all_labels_, predicted_classes)
    
#     print(f"--->>> Predict label {predicted_classes}")
#     print(f"--->>> true label {all_labels_}")
    

#     return {
#         "accuracy": accuracy,
#         "Precision": precision,
#         "F1-Score": f1_c,
#         "recall" : recall,
#         "roc_auc" : roc_,
#         "mcc": mcc_,
#         "MRR": MRR,
#         "N@5": N5,
#         "MFR": MFR,
#     }


# # model = LitGNN.load_from_checkpoint(path_)
# metrics = calculate_metrics(model, data)
# dfm = pd.DataFrame([metrics])
# dfm.to_csv(f"{outputs_dir()}/evaluation_metrics.csv", index=False)
# # this code is predicting a single graph. try to find the id of the graph






# Work on GNN explainer
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig
import torch_geometric as pyg

def run_gnn_explainer_all_features(model, data):
    """
    Runs GNNExplainer to evaluate feature importance for all node features in the graph.
    
    :param model: The GNN model (LitGNN in this case)
    :param data_loader: The DataLoader containing graphs (DGL graph objects)
    """
    for batch in data.test_dataloader():
        graph = batch  # Your DGL graph or the data for this batch
        node_features = graph.ndata["_FUNC_EMB"]  # Node features (e.g., from CodeBERT)
        labels = graph.ndata['_VULN']  # Node labels (e.g., vulnerable or not)

        # Convert DGL graph to PyTorch Geometric's format
        edge_index = torch.stack(graph.edges())  # Extract edge indices from DGL graph

        # Define the model configuration required for the explainer
        model_config = ModelConfig(
            mode='multiclass_classification',  # The model's task type: 'classification', 'regression'
            task_level='node',  # The task type: 'node' or 'edge' classification
            return_type='log_probs'  # The type of output your model returns: 'log_prob', 'prob', or 'raw'
        )
        
        # Wrap your model into PyTorch Geometric Explainer
        explainer = Explainer(
            model=model,
            algorithm= GNNExplainer(),
            explanation_type='model',
            node_mask_type='attributes',  # Masking node features (attributes)
            edge_mask_type='object',  # Masking edges (optional)
            model_config=model_config  # Pass the model configuration here
        )

        # Apply the explainer to the whole graph (explain feature importance for all nodes)
        explanation = explainer(
            x=node_features,  # The node features
            edge_index=edge_index,  # The edge index from the DGL graph
            node_index= None  # If you want explanations for all nodes, set node_idx=None
        )
        
        # Now you can visualize the feature contributions using the explanation object
        print("Feature importance across all nodes:\n", explanation.node_mask)

        # Optional visualization (You can also visualize subgraph explanations)
        explainer.visualize_subgraph(None, edge_index, explanation.edge_mask)

        # Breaking after the first batch for demonstration purposes
        break

# Call the function after training your model
run_gnn_explainer_all_features(model, data)
