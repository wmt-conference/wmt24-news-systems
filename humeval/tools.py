import csv
import ipdb
import logging
import glob
import numpy as np
import pandas as pd
from itertools import combinations
import scipy.stats as stats
from scipy.stats import wilcoxon

import warnings
warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros")
warnings.filterwarnings("ignore", message="Sample size too small for normal approximation")



ALPHA_THRESHOLD = 0.05

langcode_mapping = {
    "eng": "en",
    "ces": "cs",
    "hin": "hi",
    "zho": "zh",
    "jpn": "ja",
    "rus": "ru",
    "ukr": "uk",
    "deu": "de",
    "isl": "is",
    "spa": "es",
}

def mqm_weights(row):
    if row['severity'] == "No-error" or "Reinterpretation" in row['category']:
        return 0
    category_errors = {
        'Non-translation!': -25,
        'Source issue': 0,
    }
    if row['category'] in category_errors:
        return category_errors[row['category']]        
    if row['severity'] == "minor":
        if 'Fluency/Punctuation' in row['category']:
            return -0.1
        return -1
    if row['severity'] == "major" or row['severity'] == "critical":
        return -5
    raise Exception(f"Unknown MQM weights {row}")


def load_mqm(filename):
    df = pd.read_csv(filename, dtype=str, sep='\t', quoting=csv.QUOTE_NONE, quotechar='')
    
    df = df.rename(columns={"rater": "user_id", 'system': 'system_id', 'globalSegId': 'segment_id', 'doc': 'doc_id'})
    # mqm annotations contained CANARY string on first line which shifts everything by one
    df['segment_id'] = df['segment_id'].astype(int)
    df['segment_id'] -= 1
    df['segment_id'] = df['segment_id'].astype(str)
    
    if "_ende." in filename:
        df['source_lang'] = "en"
        df['target_lang'] = "de"
    elif "_jazh." in filename:
        df['source_lang'] = "ja"
        df['target_lang'] = "zh"
    else:
        raise Exception(f"unknown lnaguage pair")        

    df['error_spans'] = df.apply(lambda x: f"{{'severity':{x['severity']}, 'error_type:{x['category']}'}}", axis=1)

    df['rating'] = df.apply(lambda x: mqm_weights(x), axis=1)

    aggregation = {'rating': 'sum', 'error_spans': lambda x: ', '.join(x)}
    df = df.groupby(['system_id', 'doc_id', 'segment_id', 'user_id', 'source_lang', 'target_lang'], as_index=
False).agg(aggregation)
    
    return df


def load_all_resources():
    basefolder = 'wmt24-news-systems/txt'

    sources = {}
    domains = {}
    systems = {}
    for source in glob.glob(f"{basefolder}/sources/*.txt"):
        lp = source.split("/")[-1].replace(".txt", "")
        sources[lp] = []
        with open(source) as fh:
            for line in fh:
                sources[lp].append(line.strip())

        domains[lp] = []
        with open(f"{basefolder}/documents/{lp}.docs") as fh:
            for line in fh:
                domains[lp].append(line.strip().split('\t'))

        systems[lp] = {}
        for system in glob.glob(f"{basefolder}/system-outputs/{lp}/*.txt"):
            sysname = system.split("/")[-1].replace(".txt", "")
            systems[lp][sysname] = []
            with open(system) as fh:
                for line in fh:
                    systems[lp][sysname].append(line.strip())
        
    for refs in glob.glob(f"{basefolder}/references/*.txt"):
        refname = refs.split("/")[-1].split(".")[1]
        lp = refs.split("/")[-1].split(".")[0]
        systems[lp][refname] = []
        with open(refs) as fh:
            for line in fh:
                systems[lp][refname].append(line.strip())

    all_doms = {}
    for lp in domains:
        # count number of segments for each domain
        domain_count = {}
        for domain, doc in domains[lp]:
            if domain == "canary":
                continue
            if domain not in domain_count:
                domain_count[domain] = 0
            domain_count[domain] += 1
        total_count = sum(domain_count.values())
        # divide domain count by total count and multiply by 100
        for domain in domain_count:
            domain_count[domain] = domain_count[domain] / total_count * 100
            domain_count[domain] = f"{domain_count[domain]:.1f}%"
        all_doms[lp] = domain_count
    
    doms = pd.DataFrame(all_doms)
    doms.to_latex('domain_distributions.tex', index=True)

    return sources, domains, systems
     

def attach_resources(df):
    sources, domains, systems = load_all_resources()

    df['source_segment'] = ''
    df['hypothesis_segment'] = ''
    df['domain_name'] = ''
    df['document_name'] = ''

    df.reset_index(drop=True, inplace=True)
    for index, row in df.iterrows():
        # plus one is because original data dind't have CANARY STRING on first row
        seg_id = int(row['segment_id']) + 1

        lp = row['lp']
        df.loc[index, 'source_segment'] = sources[lp][seg_id]
        df.loc[index, 'hypothesis_segment'] = systems[lp][row['system_id'].replace("_", "-")][seg_id]
        domain, doc = domains[lp][seg_id]
        df.loc[index, 'domain_name'] = domain
        df.loc[index, 'document_name'] = doc

    return df


def load_data(filename, only_paired=False, is_mqm=False, remove_qc=True):
    if is_mqm:
        df = load_mqm(filename)
    else:
    # keep everything as a string to avoid errors
        df = pd.read_csv(filename, header=None, dtype=str)
        df.columns = ['user_id', 'system_id', 'segment_id', 'segment_type', 'source_lang', 'target_lang', 'rating', 'doc_id', 'unk', 'error_spans', 'start_time', 'end_time']
        
        df['source_lang'] = df['source_lang'].apply(lambda x: langcode_mapping[x])
        df['target_lang'] = df['target_lang'].apply(lambda x: langcode_mapping[x])
        
        # ramove quality control
        if remove_qc:
            df = df[df['segment_type'] == 'TGT']
            # drop tutorial "tutorial" not a subset of string in system_id column
            df = df[~df['system_id'].str.contains('tutorial')]

            # incomplete documents are used to fill the remaining items
            df = df[~df['doc_id'].str.contains('#incomplete')]

            # #dup is used to fill the account with identical evaluation
            df = df[~df['doc_id'].str.contains('#dup')]
        
    df['wave'] = filename
    df['lp'] = df['source_lang'] + '-' + df['target_lang']
    df['annot_id'] = df['system_id'] + '-' + df['doc_id']+ '-' + df['segment_id'] + '-' + df['lp'] + '-' + df['wave']
    df['orig_segment_id'] = df['doc_id']+ '-' + df['segment_id'] + '-' + df['lp']

    if not is_mqm:
        # if there are multiple ratings for the same segment by the same user, take the latest one. This happens when user changes their mind
        df = df.sort_values('end_time', ascending=False)
        df = df.drop_duplicates(['user_id', 'annot_id'], keep='first')

    # process for ranking
    df['rating'] = df['rating'].astype(float)

    # we need the segments to be evaluated across all systems to avoid bias from additional segments
    if only_paired:
        segids = set()
        for lp in df['lp'].unique():
            subdf = df[df['lp']==lp]
            paired_segids = set(subdf['orig_segment_id'].unique())
            for system in subdf['system_id'].unique():
                subset = set(subdf[subdf['system_id']==system]['orig_segment_id'])
                # keep only the intersection
                paired_segids = paired_segids.intersection(subset)
            
            original_segment_count = len(subdf)
            paired_segment_count = len(subdf[subdf['orig_segment_id'].isin(paired_segids)])
            if original_segment_count != paired_segment_count:
                logging.info(f'For {lp} removed {100*(original_segment_count - paired_segment_count)/original_segment_count:.1f}% annotations due to not being paired across all systems in {filename}')
            segids.update(paired_segids)
        
        new_df = df[df['orig_segment_id'].isin(segids)]
        df = new_df

    return df


def weighted_wilcoxon_signed_rank_test(df, x, y, macro_avg=True):
    if not macro_avg:
        differences = (df[x] - df[y]).to_list()
        return wilcoxon(differences, alternative="greater").pvalue

    pvalues = []
    for domain in df['domain_name'].unique():
        subdf = df[df['domain_name'] == domain]
        differences = (subdf[x] - subdf[y]).to_list()
        
        pvalues.append(wilcoxon(differences, alternative="greater").pvalue)


    # Stouffer's Z-score Method
    z_scores = [stats.norm.ppf(1 - p) for p in pvalues]
    weights = np.ones(len(pvalues))
    combined_z_score = np.sum(weights * z_scores) / np.sqrt(np.sum(weights ** 2))
    pvalue = 1 - stats.norm.cdf(combined_z_score)
    return pvalue



def get_pvalues(df, macro_avg=True):
    systems = df['system_id'].unique()
    ranks = {}
    for system1, system2 in combinations(systems, 2):
        df_system1 = df[df['system_id'] == system1]
        df_system2 = df[df['system_id'] == system2]

        # make annot_id the index
        df_system1 = df_system1.set_index('orig_segment_id')
        df_system2 = df_system2.set_index('orig_segment_id')

        # join the columns 'rating' from both dataframes on maximal coverage
        df_system1 = df_system1[['rating', 'domain_name']]
        df_system2 = df_system2[['rating']]
        df_system1.columns = ['rating1', 'domain_name']
        df_system2.columns = ['rating2']
        df_system = df_system1.join(df_system2, how='outer')
        # drop rows with NaN values
        df_system = df_system.dropna()
        
        ranks[(system1, system2)] = weighted_wilcoxon_signed_rank_test(df_system, 'rating1', 'rating2', macro_avg=macro_avg)
        ranks[(system2, system1)] = weighted_wilcoxon_signed_rank_test(df_system, 'rating2', 'rating1', macro_avg=macro_avg)
    
    return ranks


def get_ranks(pvalues, systems):
    wins = {}
    losses = {}
    ranks = {}
    for system1 in systems:
        wins[system1] = 0
        losses[system1] = 0
        for system2 in systems:
            if system1 == system2:
                continue
            if pvalues[(system1, system2)] < ALPHA_THRESHOLD:
                wins[system1] += 1
            if pvalues[(system2, system1)] < ALPHA_THRESHOLD:
                losses[system1] += 1

        top_rank = losses[system1] + 1
        worst_rank = len(systems) - wins[system1]
        ranks[system1] = (top_rank, worst_rank)

    return ranks, wins, losses


def generate_latex_row(row, row_type=None, supported="Yes", domains=[], last_domains={}):
    sysname = row['system_id'].replace('_', '\_')
    if sysname.startswith('ref'):
        sysname = sysname.replace('ref', 'HUMAN-')
    if supported=="No":
        sysname = f"\\nonsupporting{{{sysname}}}"
    
    rank = '-'.join(map(str, row['rank'])) if isinstance(row['rank'], tuple) else ""
    autorank = row['AutoRank']
    
    if not isinstance(autorank, str):
        autorank = f"{autorank:.1f}"
    content = f"{rank} & {sysname} & {row['rating']:.1f} & {autorank}"

    if len(domains) > 0:
        for domain in domains:
            # mark scores that are out of order
            mark = ''
            if domain in last_domains and last_domains[domain] < row[f'domain_{domain}']:
                mark += "$\wr$"
            last_domains[domain] = row[f'domain_{domain}']
            
            content += f" & {mark} {row[f'domain_{domain}']:.1f}"
                
    if row_type == 'closed-system':
        content = f"\\closedtrack{{{content}}} \\\\"
    elif row_type == 'open-source':
        content =  f"\\opentrack{{{content}}} \\\\"
    else:
        content =  f"{content} \\\\"
    
    content = content.replace('nan', '-')
    return content, row['cluster'], last_domains


def generate_table(df, lp, latex_file, extended=False):
    df['AutoRank'] = df['AutoRank'].fillna('-')
    print(f'{{\\bf{{{lp}}}}}\\\\', file=latex_file)
    domains = []
    if extended:
        for column in df.columns:
            if column.startswith('domain_'):
                domains.append(column.replace('domain_', ''))
        print(f"\\begin{{tabular}}{{clcc|{len(domains)*'r'}}}", file=latex_file)
        print(f"Rank & System & Human & AutoRank & {' & '.join(domains)}\\\\", file=latex_file)
    else:
        print("\\begin{tabular}{clcc}", file=latex_file)
        print("Rank & System & Human & AutoRank \\\\", file=latex_file)
    print("\\toprule", file=latex_file)
    last_cluster = 1
    last_domains = {}
    for index, row in df.iterrows():
        latex_row, cluster, last_domains = generate_latex_row(row, row['track'], row['lp_supported'], domains, last_domains)
        if cluster > last_cluster:
            print("\\midrule", file=latex_file)
            last_cluster = cluster
        print(latex_row, file=latex_file)
    print("\\bottomrule", file=latex_file)
    print("\\end{tabular}", file=latex_file)


def generate_latex_tables(results, extended=False):
    suffix = ''
    if extended:
        suffix = '_extended'
    latex_file = open(f'generated_human_ranking{suffix}.tex', 'w')
    for lp in results:
        table = results[lp]
        if extended:
            print("\\begin{table*}", file=latex_file)
        else:
            print("\\begin{table}", file=latex_file)
        print("\\centering", file=latex_file)
        print("\\small", file=latex_file)
        generate_table(table, lp, latex_file, extended)
        if extended:
            print("\\end{table*}\n\n", file=latex_file)
        else:
            print("\\end{table}\n\n", file=latex_file)
