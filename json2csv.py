import csv
import json

with open("oneperline.csv", "w", newline='') as out_f, open("oneperline.json") as in_f:
    fieldnames = ['text', 'rating', 'length', 'turn']
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    for in_line in in_f:
        in_json = json.loads(in_line)
        text = "\n".join([('UUUUU' if i[1] == 'User' else 'AAAAA') + ' : ' + i[0] for i in in_json['dialoge']])
        writer.writerow({'text': text,
                         'rating': in_json['rating'],
                         'length': len(text),
                         'turn': len(in_json['dialoge'])
                         })
        # text = " \n ".join([i[0] for i in in_json['dialoge']])
        # writer.writerow(
        #     {'text': text, 'rating': in_json['rating'], 'length': len(text), 'turn': len(in_json['dialoge'])})
