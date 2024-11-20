import ipdb
import pandas as pd
from itertools import combinations
from collections import defaultdict


def generate_max_per_domain(results):
    max_per_domain = {}
    for lp in results.keys():
        lp_name = lp.split(' ')[0]
        if lp_name == "Czech-Ukrainian" or "German" in lp_name or "Japanese-Chinese" in lp_name:
            continue
        lp_name = lp_name.replace("-", r"$\rightarrow$").replace("English", "En.")
        max_per_domain[lp_name] = {}
        for domain in [d for d in results[lp].columns if d.startswith('domain_')]:
            domainname = domain.split('_')[1]
            max_per_domain[lp_name][domainname] = results[lp][domain].max()
    df = pd.DataFrame(max_per_domain).transpose()
    df['Average'] = df.mean(axis=1)
    # column average
    df.loc['Average'] = df.mean(axis=0)

    df.to_latex(
        'tables/max_per_lp_per_domains.tex',
        float_format="%.1f",
        column_format=r"l>{\hspace{-3mm}}rrrrr",
        escape=False,
    )

def generate_head_to_head(data):
    filehandle = open(f"tables/head_to_head.tex", 'w')

    for lp in data:
        scores, pvalues, ranks, clusters = data[lp]
        systems = scores.index

        # scores is pandas series
        table = {}
        for system in systems:
            table[system] = {}
            for system2 in scores.index:
                diff = scores[system] - scores[system2]
                if system == system2 or diff < 0:
                    table[system][system2] = "--"
                else:
                    statsymbol = ""
                    if pvalues[system, system2] < 0.001:
                        statsymbol = "$\\ddagger$"
                    elif pvalues[system, system2] < 0.01:
                        statsymbol = "$\\dagger$"
                    elif pvalues[system, system2] < 0.05:
                        statsymbol = "$\\star$"                        
                    table[system][system2] = f"{diff:.1f}{statsymbol}"
        
        
        print(r'\begin{table*}[h]', file=filehandle)
        print(r'\centering', file=filehandle) # let's keep it in center
        print('{\\bf\\small Head to head comparison for %s systems}' % lp.replace("-", r"$\rightarrow$"), file=filehandle)
        # zouharvi: for page layout alignment
        if "English-German" in lp or "English-Czech" in lp:
            print(r'\fontsize{4}{4}\selectfont', file=filehandle)
        else:
            print('\\tiny', file=filehandle)
        # Generate column specification with vertical lines between clusters
        col_spec = 'r|'
        for i in range(len(systems)):
            if i > 0 and clusters.iloc[i] != clusters.iloc[i - 1]:
                col_spec += '|'
            col_spec += 'c'
        print('\\begin{tabular}{%s}' % col_spec, file=filehandle)

        # Generate header row with rotated system names
        header_row = ' & ' + ' & '.join(['\\rotatebox{90}{%s}' % s for s in systems]) + '\\\\'
        print(header_row, file=filehandle)
        print('\\hline', file=filehandle)

        # Generate data rows with horizontal lines between clusters
        for i in range(len(systems)):
            if i > 0 and clusters[i] != clusters[i - 1]:
                print('\\hline', file=filehandle)
            row = systems[i]
            for j in range(len(systems)):
                value = table[systems[i]][systems[j]]
                row += ' & ' + value
            row += '\\\\'
            print(row, file=filehandle)

        print('\\hline', file=filehandle)        
        scores_row = 'Scores'
        for system in systems:
            score = scores.get(system, '')
            scores_row += f' & {score:.1f}'
        scores_row += '\\\\'
        print(scores_row, file=filehandle)   

        scores_row = 'Ranks'
        for system in systems:
            rank = ranks.get(system, '')
            scores_row += ' & ' + f"{rank[0]}-{rank[1]}"
        scores_row += '\\\\'
        print(scores_row, file=filehandle)
        
        print('\\end{tabular}', file=filehandle)
        print('\\end{table*}\n\n', file=filehandle)


def generate_online_llm_head_to_head_wins(head_to_head, results_extended):
    llms_head_to_head = defaultdict(lambda: {"A>B": 0, "B>A": 0, "tie": 0})
    llms = [
        'Claude-3.5', 'ONLINE-B', 'Aya23', 'Gemini-1.5-Pro', 'Llama3-70B', 'ONLINE-A', 'ONLINE-W',
        'Mistral-Large', 'GPT-4', 'CommandR-plus', 'ONLINE-G'
    ]
    for lp in head_to_head.keys():
        systems = head_to_head[lp][0].keys()
        
        for systemA, systemB in combinations(llms, 2):
            syspair = (systemA, systemB)
            # if systemA in systems and systemB not in systems:
            #     llms_head_to_head[syspair]["A>B"] += 1
            # elif systemB in systems and systemA not in systems:
            #     llms_head_to_head[syspair]["B>A"] += 1
            # elif systemA not in systems and systemB not in systems:
            #     llms_head_to_head[syspair]["tie"] += 1
            if systemA not in systems or systemB not in systems:
                if systemA not in list(results_extended[lp]['system_id']) or systemB not in list(results_extended[lp]['system_id']):
                    continue
                scoreA = results_extended[lp][results_extended[lp]['system_id']==systemA].iloc[0]['AutoRank']
                scoreB = results_extended[lp][results_extended[lp]['system_id']==systemB].iloc[0]['AutoRank']

                # AutoRank is in ascending order
                if scoreA < scoreB:
                    llms_head_to_head[syspair]["A>B"] += 1
                elif scoreB < scoreA:
                    llms_head_to_head[syspair]["B>A"] += 1
                else:
                    llms_head_to_head[syspair]["tie"] += 1
                
            else:
                if head_to_head[lp][1][syspair] <= 0.05:
                    if head_to_head[lp][0][systemA] > head_to_head[lp][0][systemB]:
                        llms_head_to_head[syspair]["A>B"] += 1
                    else:
                        llms_head_to_head[syspair]["B>A"] += 1
                else:
                    llms_head_to_head[syspair]["tie"] += 1
                

    winrates = defaultdict(dict)
    wins = defaultdict(int)
    for systemA in sorted(llms):
        for systemB in sorted(llms):
            if systemA == systemB:
                continue
            syspair = (systemA, systemB)
            if syspair not in llms_head_to_head:
                continue
            # denominator = llms_head_to_head[syspair]["A>B"] + llms_head_to_head[syspair]["B>A"] + llms_head_to_head[syspair]["tie"]
            # if denominator == 0:
            #     continue
            # winrateAB = round(100 * llms_head_to_head[syspair]["A>B"] / (denominator), 1)
            # winrateBA = round(100 * llms_head_to_head[syspair]["B>A"] / (denominator), 1)
            winrateAB = f"{llms_head_to_head[syspair]['A>B']}/{llms_head_to_head[syspair]['tie']}/{llms_head_to_head[syspair]['B>A']}"
            winrateBA = f"{llms_head_to_head[syspair]['B>A']}/{llms_head_to_head[syspair]['tie']}/{llms_head_to_head[syspair]['A>B']}"
            assert systemB not in winrates[systemA]
            winrates[systemA][systemB] = winrateAB
            assert systemA not in winrates[systemB]
            winrates[systemB][systemA] = winrateBA
            if llms_head_to_head[syspair]["A>B"] > llms_head_to_head[syspair]["B>A"]:
                wins[systemA] += 1
            elif llms_head_to_head[syspair]["B>A"] > llms_head_to_head[syspair]["A>B"]:
                wins[systemB] += 1

    df = pd.DataFrame(winrates)
    # sort columns based on wins
    df = df.reindex(sorted(df.columns, key=lambda x: wins[x], reverse=True), axis=1)


    # add column "wins"
    df['wins'] = df.index.map(lambda x: wins[x])
    # move wins to first column
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    
    # sort rows based on wins
    df = df.sort_values(by='wins', ascending=False)

    # rename all columns and wrap tehm in \rotatebox{90}{}
    df.columns = [f"\\rotatebox{{90}}{{{c}}}" for c in df.columns]

    # replace NaN with "--"


    # to latex
    df.to_latex(
        'tables/llm_online_head_to_head.tex',
        float_format="%.1f",
        column_format=r"lr|"+r"r"*(len(df.columns)-1),
        escape=False,
        na_rep="--",
    )