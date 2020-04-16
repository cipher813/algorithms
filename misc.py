import os
import re
import json
import s3fs
import glob
import boto3
import shutil
import pickle
import random
import tarfile
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from pathlib import Path
import dask.dataframe as dd
from typing import List, Dict
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from mpl_toolkits.mplot3d import Axes3D

from cloud.aws.s3.core import download, download_dir

from sklearn.metrics import (auc,precision_recall_curve, confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score, f1_score, average_precision_score)

#from renegade_banzai.metrics import ClasswiseWeightedRecall

#from renegade_model.evaluation.metrics import top_k_accuracy
#from renegade_model.indexed_file import IndexedJsonFile
# from renegade_model.evaluation.plotting import plot_confusion_matrix

# FILE MANAGEMENT
def data_generator(file):
    """template data generator."""
    for row in open(file):
        yield row

def generator(item_list: List[str]) -> str:
    """Singly iterate through items in a list; helpful to iterate 
    through large datasets that can't fit into memory.
    
    generate = generator(item_list)
    next(generate)
    """
    for item in item_list:
        yield item
        
def open_serial_file(filepath, pkl_path, filetype="csv", **kwargs):  # usecols=None, index_col=None, nrows=None
    """Open large file (from s3 or local), serializing for later use
    
    Args
    :filepath: str s3 or local path to file
    :pkl_path: Union[str,Path] filepath at which to save serialized file
    :dtype: dict of declared data types
    :filetype: str filetype to open.  Compatible formats include csv, jsonl (json lines) or json
    
    Returns pd.DataFrame dataframe    
    """
    if not os.path.exists(pkl_path):
        if filetype=="csv":
            df = pd.read_csv(filepath, **kwargs)
        elif filetype=="jsonl":
            df = pd.read_json(filepath, lines=True, **kwargs)
        elif filetype =="json":
            df = pd.read_json(filepath, **kwargs)
        else:
            raise "Incompatible filetype."
        df.to_pickle(pkl_path)
    df = pd.read_pickle(pkl_path)
    return df

def load_json(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
#    with open(path) as json_file:  
#        df = json.load(json_file)
    return data

def df_to_s3(df: pd.DataFrame, outpath: str, index:bool=True, header:bool=True, compression: str='infer') -> None:
    """
    TODO fix compression per https://github.com/pandas-dev/pandas/issues/7615
    Save dataframe directly to S3
    Args
    :df: Dataframe to save
    :outpath: s3 path destination
    """
    s3 = s3fs.S3FileSystem(anon=False)
    with s3.open(outpath, "w") as f:
        df.to_csv(f,index=index,header=header,compression=compression)
        
def download_from_s3(s3_path: str, filepath_or_dir: str) -> str:
    """Download file/directory from s3
    Args
    :s3_path: source file/directory
    :filepath_or_dir: target file/directory
    """
    if not os.path.exists(filepath_or_dir) or len(os.listdir(filepath_or_dir))<1:
        if s3_check_isfile(s3_path):
            download(s3_path, filepath_or_dir)
            print("File downloaded.")
        else:
            download_dir(s3_path,filepath_or_dir)
            print("Directory downloaded.")
    else:
        print(f"{filepath_or_dir} already exists.")
    return filepath_or_dir

def upload_directory(dir_path: str,s3_path: str) -> None:
    """
    DEPRECATED use cloud.aws.s3.core import upload
    Upload a directory/folder to S3
    Args
    :dir_path: source folder
    :s3_path: destination folder
    """
    bucket, key = re.findall(r"\/\/([^\/]+)\/(.+)", s3_path)[0]
    s3 = boto3.resource('s3')
    base = None
    for root,dirs,files in os.walk(dir_path):
        if base==None:
            base = root
        d = root.replace(base,"")
        for f in files:
            fp = os.path.join(key,d,f)
            s3.meta.client.upload_file(os.path.join(root,f),bucket,fp)
            print(f"uploaded {fp}")
            
def s3_check_isfile(path: str) -> bool:
    s3 = s3fs.S3FileSystem()
    return s3.isfile(path)

def create_directory(path: str,exist_ok: bool=False)-> str:
    if os.path.exists(path):
        if exist_ok:
            print(f"{path} exists")
            return path
        else:
            print(f"Overwriting {path}")
            shutil.rmtree(path)
    else:
        print(f"Creating path {path}")
    path = Path(path)
    path.mkdir(parents=True,exist_ok=True)
    return str(path)

def combine_extracts(extract_path: str, header: List[str], column_list: List[str]=None,dtype=None) -> pd.DataFrame: 
    """Combine extracts into a single dataframe
    Args
    :extract_path: path to extracts directory on s3
    :header: list of headers
    :column_list: list of columns to keep
    """
    # convert dtype after import using .astype
    extract_path = Path(extract_path)
    pkl_path = extract_path.parents[0] / "extracts.pkl"
    if not os.path.exists(pkl_path):
        if os.path.isfile(extract_path):
            df = pd.read_csv(extract_path,names=header,dtype=dtype)
            if column_list:
                df = df[column_list]
        else:
            df = pd.DataFrame()
            for idx, fil in enumerate(extract_path.glob("data_*")):
                df1 = pd.read_csv(os.path.join(extract_path,fil),names=header,dtype=dtype)
                if column_list:
                    df1 = df1[column_list]
                df = df.append(df1)
                print(f"Appended {idx}: {fil}",end="\r")
        df.to_pickle(pkl_path)
        print(f"Combined extract pickled at {pkl_path}")
    df = pd.read_pickle(pkl_path)
    print(f"Data loaded from {pkl_path}")
    return df

def import_and_filter_scored(inpath: str, outpath: str,k:int=1,claim_list: List[str]=None,
                             zfill:int=None,col_idx:int=3) -> pd.DataFrame:
    """Import and filter codes csv
    Args
    :inpath: path/to/codes/csv
    :outpath: save as this pickle file
    :claim_list: filter for claims in this list
    """
    if os.path.exists(outpath):
        df = pd.read_pickle(outpath)
    else:
        df = pd.DataFrame()
        for i,chunk in enumerate(pd.read_csv(inpath, chunksize=1e6,index_col=False)): # load 1m rows at a time
            if zfill:
                chunk.columns = [x.zfill(zfill) if x.isdigit() else x for x in chunk.columns]
            if claim_list:
                chunk = chunk[chunk.claim_id.isin(claim_list)]
            df1 = topcodes_to_column(chunk,k,col_idx)[['claim_id','code','score']] 
            print(i, chunk.shape, df1.shape)
            df = df.append(df1)
        df.to_pickle(outpath)
    return df

def claim_to_file(inpath: str, header: List[str], outpath: str) -> None:
    """Save formatted claim to file
    Args
    :inpath: path/to/csv
    :header: header for generator
    :outpath: Directory at which to save claim files in api json format
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    i = 0
    generator = get_claims_iterator(str(inpath),header,',')
    for claim in generator:
        claim = format_json_for_api(claim,0.9)
        cid = str(claim['claims'][0]['claimId']).split('-')[-1]
        path = os.path.join(str(outpath),f"claim_{cid}")
        with open(path, 'w') as f:
            json.dump(claim, f)
        print(f"claim {i} saved at {path}")
        i+=1
    print("All claims completed")

def prep_folder_with_api_files(score_df: pd.DataFrame, extract_df: pd.DataFrame, outpath1: str, outpath2: str, 
                               label: str, left_columns: List[str]) -> None:
    """Compile folder with api formatted json files
    Args
    :score_df: score dataframe
    :extract_df: extract dataframe
    :outpath1: interim local destination path
    :outpath2: final destination path
    :label: type of content 'ie revenue_code_missing'
    :left_columns: columns from score dataframe to keep in merged dataframe
    """
    score_df['sample_type'] = label
    df = score_df[left_columns].merge(extract_df,on="claim_id",how="left") 
    df = df.sort_values("claim_id")
    header = df.columns
    df.to_csv(outpath1,header=False,index=False)
    claim_to_file(outpath1,header, outpath2)
         
def make_directories(directories_list:List[Path],purge=False)->None:
    """Create required directories"""
    if purge and os.path.exists(directories_list[0]):
            for fp in directories_list:
                shutil.rmtree(fp)
    for fp in directories_list:
        fp.mkdir(parents=True,exist_ok=True)
        
def load_extract(s3_dir, pklpath, **kwargs):
    if not pklpath.exists():
        extract = dd.read_csv(s3_dir, **kwargs).compute()
        extract.to_pickle(pklpath)
    extract = pd.read_pickle(pklpath)
    return extract        

def copy_file_type_to_dir(src_path:str, dst_path:str, file_type="png") -> None:
    """Copies file with specific suffix to folder for downloading
    Args
    :src_path: Directory from which to extract files
    :dst_path: Directory to save files with selected suffix
    :file_type: File type to copy
    """
    make_directories([dst_path])
    for root,dirs,files in os.walk(src_path):
        for fil in files:
            if fil.split(".")[-1]=="png":
                shutil.copyfile(os.path.join(root,fil),os.path.join(dst_path,fil))
                print(f"Copied {fil} to {dst_path}")

# DATA TRANSFORM
def create_mapping(dataframe: pd.DataFrame, mapping:Dict[str,str]):
    """Create a dictionary mapping of dataframe columns
    Args
    :dataframe: Dataframe with columns to map
    :mapping: Dictionary with key columns to map to value columns
    """
    d = defaultdict(str)
    for k,v in mapping.items():
        for i in range(len(dataframe)):
            d[dataframe[k].iloc[i]] = dataframe[v].iloc[i]
    return dict(d)

def split_drg_and_revgroup(df: pd.DataFrame) -> pd.DataFrame:
    """Split into drg and revgroup columns
    Args
    :df: data with column to split
    """
    def split_drg(row: str) -> str:
        return "".join(re.findall(r"A_(\d+).+",row))

    def split_revgroup(row: str) -> str:
        return "".join(re.findall(r"A_\d+_(.+)",row))
    
    df['drg'] = df['DRG_REV'].apply(split_drg)
    df['revgroup'] = df['DRG_REV'].apply(split_revgroup)
    df['Charge'] = df['Charge'].apply(lambda x: f"{x:.2f}")
    df.drop(['DRG_REV','QUANTITY'],axis=1, inplace=True)
    return df

def topcodes_to_column(df: pd.DataFrame,k: int, col_idx: int=1) -> pd.DataFrame:
    """Convert code columns to a single column with top k codes
    Args
    :df: dataframe which contains codes
    :k: return top k codes for each row
    :col_idx: code columns begin at this index
    """
    codes = df.columns[col_idx:] # not sure if codes start at index 1, but you get the point
    probs = df[codes].values
    top_k_idx_unordered = np.argpartition(probs, -k)[:, -k:]
    top_k_scores_unordered = probs[np.arange(probs.shape[0])[:, None], top_k_idx_unordered]
    sorted_top_k_idx = np.argsort(top_k_scores_unordered) # e.g. [[1, 0, 2], [2, 1, 0], ...]
    top_k_idx_ordered = top_k_idx_unordered[np.arange(probs.shape[0])[:, None], sorted_top_k_idx] # e.g. [[1001, 45, 18], ...]
    top_k_scores_ordered = probs[np.arange(probs.shape[0])[:, None], top_k_idx_ordered] # e.g. [[0.4, 0.2, 0.1], ...]
    top_k_code_names = [list(x[::-1]) for x in [codes[i] for i in top_k_idx_ordered]]
    top_k_scores = [[round(x,2) for x in scores][::-1] for scores in top_k_scores_ordered] #f"{x:.2f}"
    df['code']=top_k_code_names
    df['score'] = top_k_scores
    return df

def topcodes_to_rows(df: pd.DataFrame,k:int) -> pd.DataFrame: #, dtype: str
    """Convert column with top codes list into separate rows
    :df: dataframe with column to convert to rows
    :dtype: 'Rev' or 'Proc'
    """
    data = [[i[1], col2] for i in df.itertuples() for col2 in i[2]]
    df = pd.DataFrame(data =data, columns=df.columns)
    length = len(data)//k
    df['rank'] = list(range(1,k+1)) * length
    return df

def col_to_float(df, column_list):
    for column in column_list:
        df[column] = df[column].astype(float)
    return df

def generate_drg_list(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare list of drg by claim.
    Args
    :df: extract data
    """
    df = df[['claim_id','diagnosis_related_group']].drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return df

# ANALYTICS
def count_drg_claims(df):
    df = df[df['diagnosis_related_group'].notnull()]
    df = df.drop_duplicates(['claim_id','diagnosis_related_group'])
    print(f"There are {len(df)} unique claims with DRG codes")
    return df

def drg_stats(df, category):
    df = df[df['category']==category]
    drg = df[df['diagnosis_related_group'].notnull()]
    pct = 100*(len(drg)/len(df)) if len(df) else 0
    print(f"{pct:.2f}% ({len(drg)}/{len(df)}) of {category} contain DRG codes")

def threshold_percent_of_total(df: pd.DataFrame, category_list: pd.DataFrame, category:str, threshold:float) -> pd.DataFrame:
    """Calculate the ratio of rows over threshold vs total rows in specified category
    Args
    :df: binary scored dataframe (ie rev, proc)
    :category_list: dataframe of categories for all claims
    :category: filter by this category
    :threshold: filter 'missing' by this threshold
    """
    df = df.merge(category_list,on="claim_id",how="left")
    df1 = df[df['category']==category]
    df2 = df1[df1['missing']>threshold]
    pct = 100*(len(df2)/len(df1)) if len(df1) else 0
    print(f"{pct:.4f}% ({len(df2)}/{len(df1)}) of {category} are above a threshold of {threshold}")
    return df1
    
def oov_stats(df, oov_list, k):
    df = [c for row in df.code for c in row]
    df1 = Counter(df)
    df1 = pd.DataFrame(df1,index=[0]).T.sort_values(0,ascending=False)
    df2 = df1[df1.index.isin(oov_list)]
    print(f"OOV makes up {100*(sum(df2[0])/len(df)):.5f}% ({sum(df2[0])}/{len(df)}) of top {k} codes.")
    return df2
        
# def format_json_for_api(claim: dict, precision_threshold: float):
#     request = {"precisionThreshold": precision_threshold, "claims": []}
#     request["claims"].append(claim_to_request(claim))
#     return request

def peek(df: pd.DataFrame, columns: List[str]=None)->pd.DataFrame:
    """Look at contents of dataframe.  Like .head(), but more informative"""
    print(df.shape)
    print(df.columns)
    if columns:
        print("\nUnique Values:")
        for column in columns:
            num_unique = df[column].nunique()
            print(f"{column}: {num_unique}")
    return df.head()

def weighted_average_row(dataframe,weight_col):
    dataframe.loc['Weighted Average'] = None
    for col in dataframe.columns:
        dataframe.loc["Weighted Average",col] = np.sum((dataframe[col]*dataframe[weight_col])/dataframe[weight_col].sum())
    return dataframe

def calc_pct_of_category(dataframe,group,category):
    df = dataframe.groupby([group,category]).agg({"claim_id":"count"})
    df = df.groupby(level=0).apply(lambda x: x / float(x.sum()))
    df = df.unstack()
    df = df.fillna(0)
    df.columns = [f"pct_{x.lower().replace(' institutional ','_')}" for x in df.columns.get_level_values(1)]
    return df.round(2)

def open_image(filepath):
    """Open image artifact from path"""
    im = Image.open(filepath)
    display(im)
                
# PLOTTING
def correlation_performance_vs_attr(dataframe,attr_list, title,savepath):
    corr = dataframe[attr_list].corr()
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(16, 16))
    fig = sns.heatmap(corr, cmap=cmap, center=0,annot=True,fmt=".2f",
                square=True, linewidths=.5, cbar_kws={"shrink": .5}).get_figure()
    plt.title(title)
    fig.savefig(savepath,bbox_inches="tight")
    
def plot_performance_hist(dataframe,title, savepath):
    fig = plt.figure()
    ax = dataframe[[x for x in dataframe.columns if 'recall_95' in x]].plot.hist(figsize=(20,8),density=True, alpha=0.3)
    ax.set_title(title)
    fig = ax.get_figure()
    fig.savefig(savepath,bbox_inches="tight")

def plot_results(df,title,savepath,fontsize=10, figsize=(20,8),coords=(0.04,-0.02),ylabel="percent"):
    fig = plt.figure()
    ax = df.plot.bar(title=title,figsize=figsize,width=0.8)
    ax.set_ylabel(ylabel)
    for p in ax.patches:
        ax.annotate(str(p.get_height().round(3)), (p.get_x()+coords[0], p.get_height()+coords[1]),rotation=90, fontsize=fontsize)
    fig = ax.get_figure()
    fig.savefig(savepath,bbox_inches="tight")
    
def plot_histogram(df,column,title,savepath):
    """Plots and saves histogram"""
    plt.subplots(figsize=(20,8))
    fig = sns.distplot(df[column], kde=False).set_title(title)
    fig.figure.savefig(savepath,bbox_inches="tight")
    
def plot_histograms(dataframe,metric_list,title,label, savedir):
    for metric in metric_list:
        plot_histogram(dataframe,metric,f"{title}: {metric}",savedir / f"{label}_{metric}_histogram.png")

def line_volume_vs_performance(score_df,metric,title,savepath,top_x=None):
    """Dual axis line volume vs performance metric"""
    top_x==len(score_df) if top_x is None else top_x
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
#    df2 = score_df.groupby(grouping).agg({"volume":"sum"}).sort_values("volume",ascending=False)
    score_df['volume'].iloc[:top_x].plot.bar(figsize=(20,8), title=title,ax=ax).get_figure()
    ax.set_ylabel("Line Volume")
    score_df[metric].iloc[:top_x].plot.line(figsize=(20,8),ax=ax2,color='orange').get_figure()
    score_df["pct_pro"].iloc[:top_x].plot.line(figsize=(20,8),ax=ax2,color='gray',marker="o",linewidth=0).get_figure()    
    ax2.set_ylabel("Percentage")
    ax2.legend()
    fig.savefig(savepath,bbox_inches="tight")
    return score_df[['volume',metric]].iloc[:top_x,:]        

def stack_bar_by_segment(dataframe, seg1,seg2,title,savepath,top_x=20):
    """Generate bar plot by seg1 (payer_id etc) as a percentage of seg2 (category etc)"""
    # multilevel groupby transformation
    df = pd.DataFrame(dataframe.groupby([seg1,seg2])['claim_id'].count())
    df = df.unstack()
    df = df.fillna(0)
    df['total'] = df.sum(axis=1)
    df = df.sort_values("total",ascending=False)
    top = df.index[:top_x]
    df = df.iloc[:,:-1].div(df['total'],axis=0)
    df = df.stack()
    df = df[(df.T != 0).any()]

    top_df = df[df.index.get_level_values(0).isin(top)]
    
    # stacked barchart
    fig, ax = plt.subplots()
    fig = top_df.unstack(level=1).plot(kind='bar',stacked=True,figsize=(20,8),title=title, ax=ax)
    ax.legend(top_df.unstack().columns.get_level_values(1))
    ax.set_ylabel("% Lines")
    fig.figure.savefig(savepath,bbox_inches="tight")
    return df

def plot_histogram(label,outpath, scores,bins):
    plt.hist(scores,30)
    plt.title(f"{label}: missing prediction distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.savefig(outpath)
    plt.show()
    
# SCORING
def capture_rate(recalls, precisions ,min_precision=0.95):
    """Calculate capture_rate"""
    capture_rate = recalls[precisions>=min_precision][0]
    return capture_rate

def average_recall_score(recalls, precisions, min_precision=0.8):
    """Calculate average_recall_score"""
    average_recall_score = recalls[precisions >= min_precision].mean()
    return average_recall_score

def pr_auc_score(recalls, precisions):
    """Calculate precision recall auc score"""
    pr_auc_score = auc(recalls,precisions)
    return pr_auc_score    

def pr_partial_auc_score(recalls, precisions, min_precision=0.8):
    """Calculate partial precision recall auc score"""
    pr_partial_auc_score = auc(recalls[precisions >= min_precision],
                               precisions[precisions >= min_precision])
    return pr_partial_auc_score

#def capture_rate_threshold_pct(y_test,y_pred,min_threshold=0.2):
#    return recall[recall.capture_rate >= min_threshold].shape[0]/self.recall.shape[0]

def scored_metrics(y_test,y_pred):
    """Prepare metrics dictionary"""
    d = {}
    precisions, recalls, thresholds = precision_recall_curve(y_test,y_pred)
#    d['acc'] = accuracy_score(y_test, y_pred)
    d['avg_prec'] = average_precision_score(y_test, y_pred)
#    d['f1'] = f1_score(y_test, y_pred)
    try:
        d['threshold_80'] = thresholds[precisions[:-1] >= 0.80][0]
    except:
        pass
    try:
        d['threshold_90'] = thresholds[precisions[:-1] >= 0.90][0]
    except:
        pass
    try:
        d['threshold_95'] = thresholds[precisions[:-1] >= 0.95][0]
    except:
        pass
    try:
        d['auc'] = roc_auc_score(y_test, y_pred)
    except ValueError:
        pass
    try:
        d['aupr'] = pr_auc_score(recalls, precisions)
    except ValueError:
        pass
    try:
        d['aupr_p'] = pr_partial_auc_score(recalls, precisions)
    except ValueError:
        pass
    try:
        d['recall_avg_80'] = average_recall_score(recalls, precisions)
    except ValueError:
        pass
    try:
        d['recall_60'] = capture_rate(recalls, precisions,0.6)
    except ValueError:
        pass
    try:
        d['recall_70'] = capture_rate(recalls, precisions,0.7)
    except ValueError:
        pass
    try:
        d['recall_80'] = capture_rate(recalls, precisions,0.8)
    except ValueError:
        pass
    try:
        d['recall_90'] = capture_rate(recalls, precisions,0.9)
    except ValueError:
        pass
    try:
        d['recall_95'] = capture_rate(recalls, precisions,0.95)
    except ValueError:
        pass
    try:
        d['recall_98'] = capture_rate(recalls, precisions,0.98)  
    except ValueError:
        pass

    return d                

def score_by_segment(dataframe,segment,target,pklpath,average=False, wtd_average=False, total=False):
    """TODO develop for multiclass
    Run score metrics for all segments
    """
    d = {}
    if not pklpath.exists():
        segment_list = dataframe[segment].unique()
        print(f"There are {len(segment_list)} unique segments.")
        dataframe['y_test'] = [1 if x=="Denial" else 0 for x in dataframe[f'{target}_true']]

        for seg, df in dataframe.groupby(segment):
            y_test = df['y_test']
            y_pred = df[f'{target}_pred_Denial']
            d[seg] = scored_metrics(y_test,y_pred)
        results = pd.DataFrame(d).T
        results.fillna(0,inplace=True)
        
        average_calc = results.mean()
        total_calc = results.sum()
        if wtd_average:
            results = weighted_average_row(results,'volume')
        if average:
            results.loc["Average"] = average_calc
        if total:
            results.loc["Total"] = total_calc
        results.to_pickle(pklpath)
        print(f"Score pickled at {pklpath}")
    results = pd.read_pickle(pklpath)
    return results

def score_by_class(dataframe, target, pred_prefix, pklpath, class_list=None):
    """Run score metrics for all classes

    Returns results and non-null micro average area under the pr curve
    """
    if not pklpath.exists():
        d = {}
        df = pd.get_dummies(dataframe,columns=[target])
        if class_list:
            classes = class_list
        else:
            classes = dataframe[target].unique()
        for c in classes:
            y_test = df[f'{target}_{c}']
            y_pred = df[f'{pred_prefix}_{c}']
            d[c] = scored_metrics(y_test, y_pred)
            d[c]['volume'] = np.sum(y_test)
        results = pd.DataFrame(d).T
        results.fillna(0,inplace=True)
        results.to_pickle(pklpath)
    results = pd.read_pickle(pklpath)        
    aupr, nnmaupr = non_null_micro_avg_aupr(results)

    return results, aupr, nnmaupr

def metric_weighted_avg(dataframe, metric, weight):
    """Calculated weighted average of metric
    Args
    :dataframe: dataframe containing features
    :metric: metric of which to calculate weighted average
    :weight: feature to use as weight (ie volume, value)
    """
    dataframe = dataframe[dataframe[metric].notnull()]
    dataframe['weight'] = dataframe[weight]/dataframe[weight].sum()
    dataframe['total'] = np.multiply(dataframe[metric],dataframe['weight'])
    wtd_metric = np.sum(dataframe['total'])
    return wtd_metric

def weighted_aupr(dataframe):
    dataframe['weight'] = dataframe['volume']/dataframe['volume'].sum()
    dataframe['total'] = np.multiply(dataframe['aupr'],dataframe['weight'])
    wtd_aupr = np.sum(dataframe['total'])
    return wtd_aupr

def non_null_micro_avg_aupr(dataframe):
    """Calculate non-null micro-average area under the pr curve
    Args
    :dataframe: dataframe generated as metrics by class with aupr and volume columns
    """
    wtd_aupr = weighted_aupr(dataframe)
    df = dataframe.drop(index="__null__")
    non_null_aupr = weighted_aupr(df)
    return wtd_aupr, non_null_aupr

def total_code_value(dataframe, target='claim_denial_code'):
    """Total potential value of all codes if 100% predicted
    
    Args:
    :dataframe: pd.DataFrame test dataset (ie test.json)
    """
    dataframe = dataframe[dataframe[target]!="__null__"]
    dataframe['sum_service_line_charge_amount'] = dataframe['service_line_charge_amount'].apply(lambda x: np.sum(x))
    total_value = np.sum(dataframe['sum_service_line_charge_amount'])
    return total_value

class BanzaiModel:
    def __init__(self, dictionary, key, base_dir, dtype=None):
        self.key = key
        self.s3 = dictionary[self.key]['s3']
        self.local = base_dir / self.key
        self.dtype = dtype
        self.target = dictionary[self.key]['target']
        self.level = self.target.split('_')[0]
        self.coverage_path = os.path.join(self.s3, "build/hooks/coverage.csv")
        
        try:
            self.classwise_metrics = pd.read_csv(os.path.join(self.s3, "build/hooks/classwise_metrics.csv"),index_col=[0])
        except:
            pass
        try:
            self.global_metrics = pd.read_csv(os.path.join(self.s3, "build/hooks/global_metrics.csv"),header=None)
        except:
            pass
        
        self.temp = Path(self.local / "temp")
        self.temp.mkdir(parents=True, exist_ok=True)
        if not 'score_path' in dictionary[self.key].keys():
            self.score_path = os.path.join(self.s3, "build/hooks/score.csv")
        else:
            self.score_path = dictionary[self.key]['score_path']
            
    def score(self, nrows=None):
        """Open score.csv, save to serialized file, and transform"""
        df = open_serial_file(self.score_path, self.temp / f"{self.key}_score.pkl", dtype=self.dtype, nrows=nrows)
        df.drop(columns=['Unnamed: 0'],inplace=True)
        if self.level=='claim':
            df['service_line_charge_amount'] = df['service_line_charge_amount'].apply(eval)
            df['sum_service_line_charge_amount'] = df['service_line_charge_amount'].apply(lambda x: np.sum([i for i in x if i is not None]))
        else: # line level 
            df['sum_service_line_charge_amount'] = df['service_line_charge_amount']            
        labels = pd.get_dummies(df[self.target])
        score_df = pd.concat([df,labels],axis=1)
        return score_df
    
    def coverage(self):
        """Open coverage.csv, save to serialized file"""
        df = open_serial_file(self.coverage_path, self.temp / f"{self.key}_coverage.pkl", dtype=self.dtype)
        return df
    
    def value_by_code(self, min_precision=0.95):
        """Calculate correct predicted value of each code"""
        d = {} 
        dataframe = self.score()
        dataframe = dataframe[dataframe[self.target]!="__null__"]

        code_list = [x for x in dataframe[self.target].unique() if x != "__null__"]

        serial_file = self.temp / f"{self.key}_{str(min_precision).split('.')[-1]}_value_by_code.pkl"
        if not os.path.exists(serial_file):
            for code in code_list:
                y_test = dataframe[code]
                y_pred = dataframe[f'classifier_prob_{code}']
                prec, rec, thrs = precision_recall_curve(y_test,y_pred)
                try:
                    thr = thrs[prec[:-1]>=min_precision][0]
                    predicted = dataframe[dataframe[f'classifier_prob_{code}']>=thr]
                    value = np.sum(predicted.sum_service_line_charge_amount)
                    d[code] = value
                except:
    #                print(f"min_precision of {min_precision} not reached for code {code}")
                    d[code] = 0

            with open(serial_file, 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(serial_file, 'rb') as handle:
            d = pickle.load(handle)
        return d
    
#    def coverage(self, total_code_value, min_precision=0.95):
#        """Value of total predicted codes over total potential value"""
#        predicted_code_values = np.sum(list(self.value_by_code(min_precision=min_precision).values()))
#        return  predicted_code_values / total_code_value
    
#    def coverage_by_precision(self):
#        df = ClasswiseWeightedRecall(self.target, "sum_service_line_charge_amount").compute(self.score())
#        totals = df[df["class"] != "__null__"].groupby("precision").sum()
#        coverage_by_precision = totals["sum_service_line_charge_amount_weighted_recall_raw"] / totals["sum_service_line_charge_amount_sum"]
#        return pd.DataFrame(coverage_by_precision)

class Metric:
    def __init__(self,local_path:Path,key:str,info_dict:Dict[str,Dict[str,str]],dtypes:Dict[str,str],score_idx:int=0,recall_idx:int=0)->None: 
        """Base class for performance metric presentation.
        Args
        :local_path: Path to save unzipped trained model directory
        :key: key to info dict
        :info_dict: stores key model data
        :dtypes: set column data types
        :score_idx: specify score artifact to use by index
        :recall_idx: specify recall artifact to use by index
        """
        self.key = key
        self.s3_path = info_dict[self.key]['s3_path']
        self.logstream = info_dict[self.key]['log']
        self.target = info_dict[self.key]['target']
        self.filter = info_dict[self.key]['filter']
        self.dtypes = dtypes
        self.local = Path(local_path) / self.key
        self.artifact = self.local / f"{self.target}/{self.filter}"
        self.temp = Path(self.local / "temp")
        self.temp.mkdir(parents=True, exist_ok=True)
        
        self._download_and_extract()
        self._add_prefix()
        
        # typically multiple score files per model.  Select with index kwarg
        self.score_paths = [str(self.local / x) for x in os.listdir(self.local) if "scored_" in x]
        
        # this assumes that the score and recall artifacts we want are always the shortest
        # which in some cases may not be true, TBD.  Logic is that the filter adds length
        self.score = open_serial_file(min(self.score_paths,key=len), self.temp / f"{self.key}_score.pkl") if len(self.score_paths)>0 else None
                
    def _download_and_extract(self):
        """Download directory from s3 and extract metrics.tar.gz
        """
        if not os.path.exists(self.local) or len(os.listdir(self.local))<1:
            download_dir(self.s3_path,self.local)
            tar = tarfile.open(os.path.join(self.local,"metrics.tar.gz"))
            tar.extractall(self.local)
            tar.close()
            
    def _add_prefix(self):
        '''add unique prefixes to all images to prevent overwriting in confluence'''
        length = len(self.key)
        path = str(self.local / f"{self.target}/{self.filter}/")
        files = os.listdir(path)
        [os.rename(os.path.join(path,fil),os.path.join(path,f"{self.key}_{fil}")) for fil in files \
         if fil.split("_")[0] != self.key and fil[-3:]=="png"]
#        print(f"PNG files now named:\n{[x for x in os.listdir(path) if x[-3]=='png']}")
    
    def open_scored(self,score_idx:int)->pd.DataFrame:
        """Open score file by index
        Args
        :idx: open score file by this index
        """
        try:
            print(f"Opening {self.score_paths[score_idx]}")
            df = pd.read_csv(self.local / self.score_paths[score_idx])
            return df
        except Exception as e:
            print(f"{e}: File does not exist.")
            
    def open_recall(self,recall_idx:int)->pd.DataFrame:
        """Open score file by index
        Args
        :idx: open score file by this index
        """
        try:
            print(f"Opening {self.recall_paths[recall_idx]}")
            df = pd.read_csv(self.local / self.recall_paths[recall_idx])
            return df
        except Exception as e:
            print(f"{e}: File does not exist.")
    
    def sagemaker_plot(self):
        """Plot sagemaker metrics"""
        train_metrics, val_metrics = get_sagemaker_metrics(self.logstream, "preprod")
        epochs = range(len(train_metrics))
        plot_loss(train_metrics, val_metrics, epochs)
        
    def get_runtime(self):
        """Get Sagemaker model runtime"""
        get_runtime(self.logstream)   
    
    def f1_macro(self):
        f1_macro = self.classification_metrics[self.classification_metrics['metric']=="f1_macro"]['value']
        return f1_macro
        
    def run_top_k_accuracy(self):
        """TODO: FIX"""
        d = {}
        d['1'] = top_k_accuracy(self.y_test,self.y_pred,1)
        d['2'] = top_k_accuracy(self.y_test,self.y_pred,2)
        d['3'] = top_k_accuracy(self.y_test,self.y_pred,3)
        d['5'] = top_k_accuracy(self.y_test,self.y_pred,5)
        return d
    
    def scored_metrics(self):
        d = scored_metrics(self.y_test, self.y_pred)
        return d

    def plot_null_pr_curve(self, label=None):
        """Plot PR Curve for NULL CLASS ONLY"""
        y_true = [0 if x=="__null__" else 1 for x in self.score.clean_claim_codes_true]
        y_pred = 1 - self.score['clean_claim_codes_pred___null__']
        prec, rec, _ = precision_recall_curve(y_true,y_pred,pos_label=1)
        plt.step(rec, prec,label=label)
        plt.legend()
        aupr = auc(rec,prec)
        return aupr
    
    def score_by_class(self, pklpath):
        """Run score metrics for all classes
        
        Returns results and non-null micro average area under the pr curve
        """
        results, aupr, nnaupr = score_by_class(self.score, self.target, f"{self.target}_pred", pklpath)
            
        return results, aupr, nnaupr

class Classification(Metric):
    def __init__(self,*args,**kwargs):
        """Metrics for classification models"""
        super(Classification,self).__init__(*args,**kwargs)
        self.y_pred = self.score[f'{self.target}_pred_argmax'] if f"{self.target}_pred_argmax" in self.score.columns else self.score[f'{self.target}_pred_Denial']
        self.y_test = self.score[f'{self.target}_true']
        self.classes = [x.split('_')[-1] for x in self.score.columns if f"{self.target}_" in x][2:]
        self.recall_paths = [str(self.local / x) for x in os.listdir(self.local) if "recall_" in x]
        self.recall = pd.read_csv(min(self.recall_paths,key=len)) if len(self.recall_paths)>0 else None     
        self.classification_metrics = pd.read_csv(self.artifact / "classification_metrics.csv")
        self.class_metrics = pd.read_csv(self.artifact / "class_metrics.csv")
        self.capture_rates = pd.read_csv(self.artifact / "capture_rates.csv")
        
    def run_confusion_matrix(self,normalize=False,figsize=(10,10)):
        cm = confusion_matrix(self.y_test,self.y_pred)
        n_classes = len(self.classes)
        figsize = (n_classes,n_classes) if np.prod([n_classes, n_classes]) > np.prod(np.array(figsize)) else figsize        
        plt.figure(figsize=figsize)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            ax = sns.heatmap(cm.round(2),cmap=plt.cm.Blues,annot=True,xticklabels=self.classes,yticklabels=self.classes,square=True)
            ax.set(xlabel='Predicted Label', ylabel='True Label')            
            sns_plot = ax.get_figure()
            sns_plot.savefig(self.local / f"{self.target}/{self.filter}/{self.key}_confusion_matrix_normalized.png",bbox_inches="tight")
        else:
            ax = sns.heatmap(cm,cmap=plt.cm.Blues,annot=True,xticklabels=self.classes,yticklabels=self.classes,fmt="d",square=True)  
            ax.set(xlabel='Predicted Label', ylabel='True Label')            
            sns_plot = ax.get_figure()
            sns_plot.savefig(self.local / f"{self.target}/{self.filter}/{self.key}_confusion_matrix_raw.png",bbox_inches="tight")            

class Regression(Metric):
    def __init__(self,*args,**kwargs):
        """Metrics for regression models"""
        super(Regression,self).__init__(*args,**kwargs)
        self.y_pred = self.score[f'{self.target}_pred']
        self.y_test = self.score[f'{self.target}_true']
        self.reg_metrics = pd.read_csv(self.local / f"{self.target}/{self.filter}/regression_metrics.csv")
        self.bin_metrics = pd.read_csv(self.local / f"{self.target}/{self.filter}/binned_metrics.csv")
        self.bins = self.bin_metrics['class']
        
    def run_confusion_matrix(self,normalize=False,figsize=(10,10)):
        target_classes = np.digitize(self.y_test, self.bins) - 1
        predicted_classes = np.digitize(self.y_pred, self.bins) - 1
        class_names = [str(bin) for bin in self.bins]      
        n_classes = len(class_names)
        figsize = (n_classes,n_classes) if np.prod([n_classes, n_classes]) > np.prod(np.array(figsize)) else figsize        
        plt.figure(figsize=figsize)
        cm = confusion_matrix(target_classes, predicted_classes, labels=np.arange(len(self.bins)))
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            ax = sns.heatmap(cm.round(2),cmap=plt.cm.Blues,annot=True,xticklabels=class_names,yticklabels=class_names,square=True)
            ax.set(xlabel='Predicted Label', ylabel='True Label')            
            sns_plot = ax.get_figure()
            sns_plot.savefig(self.local / f"{self.target}/{self.filter}/{self.key}_confusion_matrix_normalized.png",bbox_inches="tight")
        else:
            ax = sns.heatmap(cm,cmap=plt.cm.Blues,annot=True,xticklabels=class_names,yticklabels=class_names, fmt="d",square=True)
            ax.set(xlabel='Predicted Label', ylabel='True Label')            
            sns_plot = ax.get_figure()
            sns_plot.savefig(self.local / f"{self.target}/{self.filter}/{self.key}_confusion_matrix_raw.png",bbox_inches="tight")         
    
    def run_scatter(self):
        # TODO replace open artifact with actual computation of scatterplot (review source code)
        open_image(self.local / f"{self.target}/{self.filter}/{self.key}_scatter.png")

        

        
