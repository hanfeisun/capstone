import csv
import json

with open("oneperline.csv", "w", newline='') as out_f, open("oneperline.json") as in_f:
    fieldnames = ['text', 'rating', 'length', 'turn']
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    for in_line in in_f:
        in_json = json.loads(in_line)
        # writer.writerow({'text': " \n ".join([('hominid' if i[1] == 'User' else 'contraption') + ' : ' + i[0] for i in in_json['dialoge']]),'rating': in_json['rating']})
        text = " \n ".join([i[0] for i in in_json['dialoge']])
        writer.writerow(
            {'text': text, 'rating': in_json['rating'], 'length': len(text), 'turn': len(in_json['dialoge'])})
