import json
import csv

def get_dialog(inp, mod):
    if inp[0]=='"':
        inp = inp[1:]

    inp = inp.replace('<br />', ' ').replace('\r', ' ').replace('\n', ' ').split('</span> ')
    out = [inp[0]]
    for i in inp[1:]:
        if i[:42] == out[-1][:42]:
            out[-1] = out[-1] + '. ' + i[42:]
        elif len(i)>25:
            out.append(i)
    return out
  
def bild_data(raw, proc):
    with open(raw, 'r', encoding='utf-8') as data:
        data = csv.reader(data, delimiter='\t')
        with open(proc, 'w', encoding='utf-8') as file:
            for i, conv in enumerate(data):
                if i == 0:
                    continue
                p1 = conv[0].replace('<span class=participant_1>', '')
                p2 = conv[1].replace('<span class=participant_2>', '')
                dialog = get_dialog(conv[2][:-1], 'join')
                for i in range(1, len(dialog)):
                    context = [p[42:] for p in dialog[:i]]
                    try:
                        persona, responce = dialog[i].replace('<span class=participant_1>Пользователь 1: ', p1+'[-sep-]')\
                        .replace('<span class=participant_2>Пользователь 2: ', p2+'[-sep-]').split('[-sep-]')
                    except BaseException:
                        print(conv)
                        C='a'
                        break

                    persona = persona.replace('</span>', '').split('<br />')[:-1]
                    file.write(json.dumps({'context':context, 'responce':responce, 'persona':persona, 'label':1})+'\n')

with open('config.json', 'r') as config:
    config  = json.loads(config.read())
                    
bild_data(config['raw_data_path'], config['proc_data_path'])
