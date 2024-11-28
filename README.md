# WMT24 News Systems and Evaluations

This repository contains all primary submissions for the WMT24 general mt task, and the human evaluations.
The submissions are in the following directories

* `xml` : One xml file for each language pair, containing source, reference(s) and hypotheses
* `txt-ts` : Sources, references and hypotheses in text files, including test suites
* `txt` : Same, but without test suites

Tools for extracting the raw text from the XML can be found [here](https://github.com/wmt-conference/wmt-format-tools).

The human evaluations are in the `humaneval` directory.



The complete compiled data [with this script](https://github.com/wmt-conference/wmt24-news-systems/blob/main/humeval/merge_to_jsonl.py) can also be found in [public JSONL](https://github.com/wmt-conference/wmt24-news-systems/raw/refs/heads/main/jsonl/wmt24_esa.jsonl).
Note that this format is experimental and bound to change in the future.
Each line in the data looks as follows:
```
{
  "langs": "en-es",
  "line_id": 731,
  "src": "Initial applications to Harvard for a psychology masters were rejected, but was eventually admitted. The initial setbacks were due to Milgramm not taking any undergraduate courses in psychology at Queens College. In 1961, Milgram received a PhD in social psychology. He became an assistant professor at Yale around the same time.",
  "tgt": "Las solicitudes iniciales para una maestría en psicología en Harvard fueron rechazadas, pero finalmente fueron admitidas. Los reveses iniciales se debieron a que Milgramm no tomó ningún curso universitario de psicología en Queens College. En 1961, Milgram recibió un doctorado en psicología social. Se convirtió en profesor asistente en Yale casi al mismo tiempo.",
  "doc_id": "test-en-speech_8bbVFeTIIg8_000",
  "domain": "speech",
  "esa_spans": [
    {
      "start_i": 31,
      "end_i": 42,
      "severity": "minor",
      "error_type": null
    },
    # omitted for brevity
  ],
  "esa_score": "66",
  "system": "ONLINE-B",
  "annotator": "engspa7832",
  "speech_info": {
    "file": "_8bbVFeTIIg8_000.mp4",
    "youtube": "https://www.youtube.com/watch?v=8bbVFeTIIg8"
  }
}
```
