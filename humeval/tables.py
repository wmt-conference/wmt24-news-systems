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
