import pandas as pd
import ipdb
import os
from absl import flags, app
from tools import load_data, get_pvalues, get_ranks, load_all_resources, attach_resources
from tables import generate_max_per_domain
import matplotlib
matplotlib.use('Agg')


original_sources, _, _ = load_all_resources()

data = []
for wave in ['wave0.csv', 'wave1.csv', 'wave2.csv', 'wave3.csv']:
    data.append(load_data(wave, is_mqm=False))
df = pd.concat(data)
df = attach_resources(df)


# create folder human-scores
if not os.path.exists('human-scores'):
    os.makedirs('human-scores')

for lp in df.lp.unique():
    subdf = df[df.lp == lp]
    # save system level
    subdf.groupby('system_id').agg({'overall': 'mean'}).reset_index().to_csv(f'human-scores/{lp}.esa.sys.score', index=False, header=False, sep='\t')
    # save domain level
    domain = subdf.groupby(['system_id', 'domain_name']).agg({'overall': 'mean'}).reset_index()
    # change order of columns
    domain = domain[['domain_name', 'system_id', 'overall']]
    domain.sort_values("domain_name").to_csv(f'human-scores/{lp}.esa.domain.score', index=False, header=False, sep='\t')

    segment = subdf.groupby(['system_id', 'segment_id']).agg({'overall': 'mean'}).reset_index()
    # change segment into integer
    segment['segment_id'] = segment['segment_id'].astype(int)
    segment = segment[['segment_id', 'system_id', 'overall']]
    # add single segment for the canary string
    for system in segment.system_id.unique():
        new_row = pd.DataFrame([{'segment_id': -1, 'system_id': system, 'overall': 100.0}])
        segment = pd.concat([segment, new_row], ignore_index=True)
    segment = segment.sort_values(by=['system_id', 'segment_id'])

    with open(f'human-scores/{lp}.esa.seg.score', 'w') as f:
        for system in segment.system_id.unique():
            sysdf = segment[segment.system_id == system]
            for index in range(len(original_sources[lp])):
                row = sysdf[sysdf.segment_id == index - 1]
                if len(row) == 0:
                    value = None
                else:
                    value = row.overall.values[0]
                f.write(f"{system}\t{value}\n")



