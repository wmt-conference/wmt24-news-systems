import csv
import glob
import collections
import json
import ipdb
import pandas as pd


LANG_3_TO_2 = {
    "eng": "en",
    "deu": "de",
    "ces": "cs",
    "hrv": "hr",
    "plk": "pl",
    "rus": "ru",
    "zho": "zh",
    "jpn": "ja",
    "spa": "es",
    "hin": "hi",
    "isl": "is",
    "ukr": "uk",
}

def load_data_wmt():
    langs_all = [x.split("/")[-1].removesuffix(".txt") for x in glob.glob("../txt/sources/*.txt")]
    data_src = {}
    data_doc = {}
    data_tgt = collections.defaultdict(dict)
    for langs in langs_all:
        data_src[langs] = list(open(f"../txt/sources/{langs}.txt").read().splitlines())
        data_doc[langs] = list(open(f"../txt/documents/{langs}.docs").read().splitlines())
        data_doc[langs] = [x.split("\t") for x in data_doc[langs]]

        for system_f in list(glob.glob(f"../txt/system-outputs/{langs}/*.txt"))+list(glob.glob(f"../txt/references/{langs}.*.txt")):
            system = system_f.split("/")[-1].removesuffix(".txt")
            system = system.removeprefix(langs + ".")
            data_tgt[langs][system] = list(open(system_f).read().splitlines())

    def load_line(line):
        (
            user_id,
            system,
            line_id,
            tgt_type,
            lang1,
            lang2,
            esa_score,
            document_id,
            _unknown,
            esa_spans,
            _time_start,
            _time_end,
        ) = line

        if tgt_type == "BAD":
            return None
        if document_id.endswith("#bad"):
            return None
        if document_id.endswith("#dup"):
            return None
        if document_id.endswith("#incomplete"):
            return None
        if "-tutorial" in document_id:
            return None
        
        
        langs = f"{LANG_3_TO_2[lang1]}-{LANG_3_TO_2[lang2]}"
        # # original data is off by one due to initial canary line
        # line_id = int(line_id)+1
        line_id = int(line_id)

        # make sure the document name matches
        assert data_doc[langs][line_id][1] == document_id

        speech_info = None
        if "-speech_" in document_id:
            _, video_id = document_id.split("-speech_")
            youtube_id = "_".join(video_id.split("_")[:-1])
            speech_info = {
                "file": f"{video_id}.mp4",
                "youtube": f"https://www.youtube.com/watch?v={youtube_id}",
            }
        
        return {
            "langs": langs,
            "line_id": line_id,
            "src": data_src[langs][line_id],
            "tgt": data_tgt[langs][system][line_id],
            "doc_id": document_id,
            "domain": data_doc[langs][line_id][0],
            "esa_spans": json.loads(esa_spans),
            "esa_score": esa_score,
            "times": [_time_start, _time_end],
            "system": system,
            "annotator": user_id,
            "speech_info": speech_info,
        }
        
    data = []
    for wave in range(4):
        for line in csv.reader(open(f'esa_generalMT2024_wave{wave}.csv')):
            data.append(load_line(line))

    # remove Nones
    data = [x for x in data if x]

    print("Loaded", len(data), "annotation lines")

    return data

def convert_to_unified_format(data, filename):
    # load file annotator_mapping.json
    with open("annotator_mapping.json") as fh:
        annot_map = json.load(fh)
    ann_map = {v: k for k, vs in annot_map.items() for v in vs}



    df = pd.DataFrame(data)
    # each segment in final output contains f"{doc_id}_#_{line_id}"
    df['doc_id'] = df.apply(lambda x: f"{x['doc_id']}_#_{x['line_id']}", axis=1)
    del df['line_id']

    data = []
    for doc_id, group in df.groupby(by='doc_id'):
        src_text = group['src'].unique()
        assert len(src_text) == 1, "There are differences in source text"
        src_text = src_text[0]

        # sort group by esa_score so better system is first
        group = group.sort_values(by="esa_score")

        tgt_text = group[['tgt', 'system']].drop_duplicates().set_index('system').to_dict()['tgt']

        scores = {}
        for systemid, sys_group in group.groupby(by='system'):
            human_scores = []
            for _, row in sys_group.iterrows():
                human_scores.append({
                    "score": float(row['esa_score']),
                    "annotator": ann_map[row['annotator']],
                    "errors": row['esa_spans'],
                    "times": row['times'],
                })
            scores[systemid] = human_scores
            

        data.append(
            {
                "scores": scores,
                "src_text": src_text,
                "tgt_text": tgt_text,
                "doc_id": doc_id
            }
        )
        
    df = pd.DataFrame(data)
    df.to_json(filename, force_ascii=False, lines=True, orient="records")


if __name__ == '__main__':
    data = load_data_wmt()
    
    convert_to_unified_format(data, "../jsonl/wmt24-genmt-humeval.jsonl")

    with open("../jsonl/wmt24_esa_original.jsonl", "w") as f:
        f.write("\n".join([json.dumps(line, ensure_ascii=False) for line in data]))