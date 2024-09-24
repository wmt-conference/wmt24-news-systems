import ipdb
import pandas as pd


def generate_max_per_domain(results):
    max_per_domain = {}
    for lp in results.keys():
        lp_name = lp.split(' ')[0]
        if lp_name == "Czech-Ukrainian" or "German" in lp_name or "Japanese-Chinese" in lp_name:
            continue
        max_per_domain[lp_name] = {}
        for domain in [d for d in results[lp].columns if d.startswith('domain_')]:
            domainname = domain.split('_')[1]
            max_per_domain[lp_name][domainname] = results[lp][domain].max()
    df = pd.DataFrame(max_per_domain).transpose()
    df['Average'] = df.mean(axis=1)
    # column average
    df.loc['Average'] = df.mean(axis=0)

    df.to_latex('tables/max_per_lp_per_domains.tex', float_format="%.1f")


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
                if system == system2:# or diff < 0:
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
        

        print('\\begin{table*}[h]', file=filehandle)
        print('\\centering', file=filehandle)
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
        print('\\vspace{5mm}', file=filehandle)
        print('\\caption{Head to head comparison for %s systems}' % lp, file=filehandle)
        print('\\end{table*}\n\n', file=filehandle)
