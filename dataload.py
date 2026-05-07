import pandas as pd
import numpy as np
from datasets import Dataset
import re

def mytok(seq, kmer_len, s):
    seq = seq.upper().replace("T", "U")    
    kmer_list = []
    for j in range(0, (len(seq)-kmer_len)+1, s):
        kmer_list.append(seq[j:j+kmer_len])
    return kmer_list

########### loading dp dataset
def build_dp_dataset():
    def load_dataset(data_path, splits):
        df = pd.read_csv(data_path)
        df = df[df["split"].isin(splits)]
        df = df.dropna(subset=["y"])
        
            
        utr5 = df["UTR5"].values.tolist()
        utr3 = df["UTR3"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["y"].values.tolist()
        
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))        
        
        assert len(seqs) == len(ys)
        
        return seqs, ys
    
    train_seqs, train_ys = load_dataset("data/translation_rate.csv", [1,2,3])
    valid_seqs, valid_ys = load_dataset("data/translation_rate.csv", [4])
    test_seqs, test_ys = load_dataset("data/translation_rate.csv", [5])
        
    

    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test

def build_class_dataset():
    def load_dataset(data_path, splits):
        df = pd.read_csv(data_path)
        df = df[df["split"].isin(splits)]

        utr5 = df["5' UTR"].values.tolist()
        utr3 = df["3' UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["ClassificationID"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
    
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/protein_expression_5class.csv", [1,2,3])
    valid_seqs, valid_ys = load_dataset("data/protein_expression_5class.csv", [4])
    test_seqs, test_ys = load_dataset("data/protein_expression_5class.csv", [5])

    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test

def build_liver_dataset():
    def load_dataset(data_path, splits):
        df = pd.read_csv(data_path)
        df = df[df["split"].isin(splits)]
        df = df.dropna(subset=["y"])
        

        utr5 = df["5' UTR"].values.tolist()
        utr3 = df["3' UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        
        ys = np.log2(df["y"].values).tolist()
        
        
        # import code; code.interact(local=dict(globals(), **locals()))
        
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/transcript_expression_liver.csv", [1,2,3])
    valid_seqs, valid_ys = load_dataset("data/transcript_expression_liver.csv", [4])
    test_seqs, test_ys = load_dataset("data/transcript_expression_liver.csv", [5])


    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test

def build_saluki_dataset(cross):
    def load_dataset(data_path, splits):
        df = pd.read_csv(data_path)
        df = df.fillna('')
        df = df[df["split"].isin(splits)]
        
        df = df.dropna(subset=["y"])
            
        utr5 = df["UTR5"].values.tolist()
        utr3 = df["UTR3"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["y"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/mrna_half-life.csv", [0,1,2,3,4,5,6,7])
    valid_seqs, valid_ys = load_dataset("data/mrna_half-life.csv", [8])
    test_seqs, test_ys   = load_dataset("data/mrna_half-life.csv", [9])


    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test

def build_hek293t_dataset():
    def load_dataset(data_path):
        df = pd.read_csv(data_path)
        
        # Extract sequences using regex patterns
        def extract_sequence(row, tag):
            pattern = f"<{tag}>(.*?)</{tag}>"
            match = re.search(pattern, row["final_sequence"])
            return match.group(1) if match else ""
        
        utr5 = [extract_sequence(row, "5UTR") for _, row in df.iterrows()]
        cds = [extract_sequence(row, "CDS") for _, row in df.iterrows()]
        utr3 = [extract_sequence(row, "3UTR") for _, row in df.iterrows()]
        ys = df["final_label"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/hek293t/train.csv")
    valid_seqs, valid_ys = load_dataset("data/hek293t/val.csv")
    test_seqs, test_ys   = load_dataset("data/hek293t/test.csv")


    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])    

    return ds_train, ds_valid, ds_test

def build_rabies_dataset():
    def load_dataset(data_path):
        df = pd.read_csv(data_path)        
        
        utr5 = df["5UTR"].values.tolist()
        utr3 = df["3UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        
        ys = df["protein_expression"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/rabies_gfp/rabies/train.csv")
    valid_seqs, valid_ys = load_dataset("data/rabies_gfp/rabies/val.csv")
    test_seqs, test_ys   = load_dataset("data/rabies_gfp/rabies/test.csv")


    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])    


    # import code; code.interact(local=dict(globals(), **locals()))

    return ds_train, ds_valid, ds_test

def build_gfp_dataset():
    def load_dataset(data_path):
        df = pd.read_csv(data_path)        
        
        utr5 = df["5UTR"].values.tolist()
        utr3 = df["3UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        
        ys = np.log1p(df["protein_expression"]).values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/rabies_gfp/gfp/train.csv")
    valid_seqs, valid_ys = load_dataset("data/rabies_gfp/gfp/val.csv")
    test_seqs, test_ys   = load_dataset("data/rabies_gfp/gfp/test.csv")


    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])    

    return ds_train, ds_valid, ds_test


def build_604_dataset():
    def load_dataset(data_path):
        df = pd.read_csv(data_path)
        
        # Concatenate sequences row by row
        utr5 = [(str(row["5UTR"]) + str(row["kozak"])) for _, row in df.iterrows()]
        utr3 = df["3UTR"].values.tolist()
        cds = [(str(row["cds"]) + str(row["stop_codon"])) for _, row in df.iterrows()]
        
        ys = df["protein_expression"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    test_seqs, test_ys = load_dataset("data/实验数据/604.csv")

    ds_test = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)]) 
    
    
    # import code; code.interact(local=dict(globals(), **locals()))

    return ds_test


def build_auro_rsv_dataset():
    def load_dataset(data_path):
        df = pd.read_csv(data_path)
        
        # Concatenate sequences row by row
        utr5 = df["5UTR"].values.tolist()
        utr3 = df["3UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        
        ys = df["OD(450nm)"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    test_seqs, test_ys = load_dataset("data/实验数据/auro_rsv.csv")

    ds_test = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)]) 
    
    
    # import code; code.interact(local=dict(globals(), **locals()))

    return ds_test


def build_zheda_homo_dataset():
    def load_dataset(data_path):
        df = pd.read_csv(data_path)
        
        # import code; code.interact(local=dict(globals(), **locals()))
        
        # Concatenate sequences row by row
        utr5 = df["5UTR"].values.tolist()
        utr3 = df["3UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        
        ys = df["72h"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    test_seqs, test_ys = load_dataset("data/实验数据/zheda_homo.csv")

    ds_test = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)]) 
    

    return ds_test


def build_zheda_mouse_dataset():
    def load_dataset(data_path):
        df = pd.read_csv(data_path)
        
        # Concatenate sequences row by row
        utr5 = df["5UTR"].values.tolist()
        utr3 = df["3UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        
        ys = df["72h"].values.tolist()
        
        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))
        
        assert len(seqs) == len(ys)
        
        return seqs, ys

    test_seqs, test_ys = load_dataset("data/实验数据/zheda_mouse.csv")

    ds_test = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)]) 

    return ds_test