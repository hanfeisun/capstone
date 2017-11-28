import csv
import json
from stat_parser import Parser
parser = Parser()
def append_question(x):
    return x
    try:
        result = parser.parse(x)
        if result._label in ["SBARQ"]:
            print(x + " ?")
            return x + " ?"
        else:
            print(x)
            print(result._label)
            return x
    except TypeError:
        return x

with open("oneperline.csv", "w", newline='') as out_f, open("oneperline.json") as in_f:
    fieldnames = ['text', 'rating', 'length', 'turn']
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    for idx, in_line in enumerate(in_f):
        print(idx)
        in_json = json.loads(in_line)
        text = "\n".join([('UUUUU' if i[1] == 'User' else 'AAAAA') + ' : ' + (append_question(i[0]) if i[1] == "User" else i[0]) for i in in_json['dialoge']])
        writer.writerow({'text': text,
                         'rating': in_json['rating'],
                         'length': len(text),
                         'turn': len(in_json['dialoge'])
                         })
        # text = " \n ".join([i[0] for i in in_json['dialoge']])
        # writer.writerow(
        #     {'text': text, 'rating': in_json['rating'], 'length': len(text), 'turn': len(in_json['dialoge'])})
