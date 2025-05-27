# %%
import pandas as pd
import numpy as np
import tools


df0 = tools.attach_resources(tools.load_data("esa_generalMT2024_wave0.csv"))
df1 = tools.attach_resources(tools.load_data("esa_generalMT2024_wave1.csv"))
df2 = tools.attach_resources(tools.load_data("esa_generalMT2024_wave2.csv"))
df3 = tools.attach_resources(tools.load_data("esa_generalMT2024_wave3.csv"))

df = pd.concat([df0, df1, df2, df3], ignore_index=True)


# %%
words_per_hour = []
words_per_hour_short = []
words_per_hour_long = []

for user, df_user in df.groupby("user_id"):
    df_user.sort_values("start_time", inplace=True)

    times1 = np.array(df_user["start_time"].astype(float)[:-1])
    times2 = np.array(df_user["start_time"].astype(float)[1:])
    # if df_user["source_lang"].iloc[0] in {"zh", "ja", "ko"}:
    #     words = [len(x) for x in df_user["source_segment"]]
    # else:
    words = [len(x.split()) for x in df_user["source_segment"]]
    timesDelta = times2 - times1
    # filter out negative and too large values
    words_time = [(w, d) for w,d in zip(words, timesDelta) if d > 0 and d < 60*10]
    words_time_short = [(w, d) for w,d in zip(words, timesDelta) if d > 0 and d < 60*5 and w < 15]
    words_time_long = [(w, d) for w,d in zip(words, timesDelta) if d > 0 and d < 60*5 and w >= 15]

    wph = sum(d for _, d in words_time) / sum(w for w, _ in words_time) * 3600
    wph_short = sum(d for _, d in words_time_short) / sum(w for w, _ in words_time_short) * 3600 if words_time_short else 0
    wph_long = sum(d for _, d in words_time_long) / sum(w for w, _ in words_time_long) * 3600 if words_time_long else 0

    print(f"{user}: {wph:.0f} words/hour")
    words_per_hour.append(wph)
    words_per_hour_short.append(wph_short)
    words_per_hour_long.append(wph_long)

print(f"average: {np.average(words_per_hour):.0f} words/hour")
print(f"median:  {np.median(words_per_hour):.0f} words/hour")
print()
print(f"average short: {np.average(words_per_hour_short):.0f} words/hour")
print(f"median short:  {np.median(words_per_hour_short):.0f} words/hour")
print()
print(f"average long: {np.average(words_per_hour_long):.0f} words/hour")
print(f"median long:  {np.median(words_per_hour_long):.0f} words/hour")
