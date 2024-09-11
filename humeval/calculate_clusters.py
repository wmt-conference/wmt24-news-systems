import pandas as pd
import ipdb
import os
from absl import flags, app
from tools import load_data, get_pvalues, get_ranks, generate_latex_tables, attach_resources

flags.DEFINE_bool('micro', False, 'Calculate micro average instead of macro over domains?')
flags.DEFINE_bool('preload', False, 'Use pickle file?')
FLAGS = flags.FLAGS


language_mapping = {
    "en": "English",
    "cs": "Czech",
    "hi": "Hindi",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "de": "German",
    "is": "Icelandic",
    "es": "Spanish",
}


def main(argv):

    if FLAGS.preload and os.path.exists('data.pkl'):
        df = pd.read_pickle('data.pkl')
    else:
        data = []
        for wave in ['mqm_generalMT2024_ende.tsv', 'mqm_generalMT2024_jazh.tsv', 'wave0.csv', 'wave1.csv', 'wave2.csv', 'wave3.csv']:
            data.append(load_data(wave, is_mqm=wave.startswith("mqm")))

        # merge data as all waves has the same columns
        df = pd.concat(data)
        df = attach_resources(df)
        # save pickle
        df.to_pickle('data.pkl')
    
    # if there are multiple ratings for the same segment, average them
    subdf = df.groupby(['annot_id', 'lp', 'system_id', 'orig_segment_id'])
    if len(subdf) != len(df):
        df = subdf.agg({'overall': 'mean'}).reset_index()

    # read file AutoRank.xlsx containing multiple sheets, each as individual df
    autoranks = pd.read_excel('AutoRank.xlsx', sheet_name=None)

    total_cluster = 0
    results = {}
    results_extended = {}
    for lp in sorted(df['lp'].unique()):
        lp_name = language_mapping[lp[0:2]] + '-' + language_mapping[lp[3:5]]
        df_lp = df[df['lp'] == lp]

        avg_annotations_per_system = df_lp.groupby('system_id')['annot_id'].count().mean()
        
        # we are far from enough annotations
        if avg_annotations_per_system < 100:
            continue

        per_domain = df_lp.groupby(['system_id', 'domain_name'])['overall'].mean().reset_index()
        avg_rating = per_domain.groupby('system_id')['overall'].mean().reset_index().set_index('system_id', drop=False)
        for domain in per_domain['domain_name'].unique():
            avg_rating[f"domain_{domain}"] = per_domain[per_domain['domain_name'] == domain].set_index('system_id')['overall']
            avg_rating['position_' + domain] = avg_rating['domain_' + domain].rank(ascending=False)
        avg_rating['position_overall'] = avg_rating['overall'].rank(ascending=False)
        
        # if not macro average, overwrite avg_rating with system average
        if FLAGS.micro:
            avg_rating['overall'] = df_lp.groupby('system_id')['overall'].mean().reset_index().set_index('system_id')['overall']
        

        # sort by overall rating
        avg_rating = avg_rating.sort_values('overall', ascending=False)

        pvalues = get_pvalues(df_lp, not FLAGS.micro)
        ranks, wins, losses = get_ranks(pvalues, df_lp['system_id'].unique())

        # merge ave_rating and ranks
        avg_rating['rank'] = avg_rating['system_id'].apply(lambda x: ranks[x])
        avg_rating['wins/losses'] = avg_rating['system_id'].apply(lambda x: f"{wins[x]}/{losses[x]}")
        avg_rating['cluster'] = 0
                    
        current_cluster = 1
        max_cutoff = 1
        for index, row in avg_rating.iterrows():
            position = row['position_overall']
            if position > max_cutoff:
                current_cluster += 1
            avg_rating.at[index, 'cluster'] = current_cluster

            max_cutoff = max(max_cutoff, row['rank'][1])

        # load autoranks
        # if systemname contain space, it is additional information; update column "Unnamed: 0" to drop everything past a space
        autoranks[lp]['Unnamed: 0'] = autoranks[lp]['Unnamed: 0'].apply(lambda x: x.split(' ')[0])
        # make column "Unnamed: 0" the index
        autoranks[lp] = autoranks[lp].set_index('Unnamed: 0')
        # add AutoRank column to avg_rating scores
        def get_autorank(system_id, column='AutoRank'):
            if system_id in autoranks[lp].index:
                return autoranks[lp].loc[system_id, column]
            
            return None
            
        avg_rating['AutoRank'] = avg_rating['system_id'].apply(lambda x: get_autorank(x)).round(1)
        avg_rating['track'] = avg_rating['system_id'].apply(lambda x: get_autorank(x, 'type'))
        avg_rating['track'] = avg_rating['track'].fillna('closed-system')
        avg_rating['lp_supported'] = avg_rating['system_id'].apply(lambda x: get_autorank(x, 'lp_supported'))

        print(f"{lp}; Clusters: {avg_rating['cluster'].max()}; Average annot. per system: {avg_annotations_per_system:0.1f}")
        print(avg_rating)
        print("#"*50)

        total_cluster += avg_rating['cluster'].max()

        lp_name = f"{lp_name} ({avg_annotations_per_system:0.0f} segments per system)"
        results[lp_name] = avg_rating


        # add data not used in human evaluation
        for index, row in autoranks[lp].iterrows():
            if index in avg_rating['system_id'].values:
                continue
            result = {
                "system_id": index,
                "AutoRank": row['AutoRank'],
                "track": row['type'],
                "lp_supported": row['lp_supported'],
                "cluster": total_cluster + 1,
            }
            # append result to avg_rating
            avg_rating = pd.concat([avg_rating, pd.DataFrame([result])], ignore_index=True)

        results_extended[lp_name] = avg_rating

    print(f"Total clusters: {total_cluster}")

    generate_latex_tables(results)
    generate_latex_tables(results_extended, extended=True)


    

if __name__ == '__main__':
    app.run(main)
