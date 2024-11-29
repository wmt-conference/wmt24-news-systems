import csv

for wave in range(4):
    with open(f"old/wave{wave}.csv") as f:
        data = list(csv.reader(f))
        for line in data:
            line[2] = int(line[2]) + 1

    with open(f"esa_generalMT2024_wave{wave}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)