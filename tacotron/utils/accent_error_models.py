from dicts import *
import argparse
import gc
import os 

def strip_x(phones):
    return [ph.lower()[1:].upper() if ph.lower()[0] == 'x' else ph.upper() for ph in phones]
             
def get_substitutions(prons, l1):
    subs = []
    prns = [' '.join(strip_x(pron['phones'])).upper() for pron in prons]
    for pron in prons:
        for j in range(len(pron['phones'])):
            for sub in substitutions[l1]:
                if pron['phones'][j].upper() == sub[0]:
                    a = pron['phones'][:j]+['X'+sub[1]]+pron['phones'][j+1:]
                    if ' '.join(strip_x(a)).upper() not in prns:
                        subs += [' '.join(a)]
    return [{'phones':s.split()} for s in list(set(subs))]

def get_deletions(prons, l1):
    prns = [' '.join(strip_x(pron['phones'])).upper() for pron in prons]
    del_c = [' '.join(pron['phones'][:-1]).upper() for pron in prons if pron['phones'][-1].lower() in arpabet_consonants and len(pron['phones']) > 1] if l1 in final_deletion and final_deletion[l1] else []
    dels = [s for s in del_c if ' '.join(strip_x(s.split())).upper() not in prns]
    for pron in prons:
        if l1 in cluster_reduction and cluster_reduction[l1]:
            del_c = [' '.join(pron['phones'][:j] + pron['phones'][j+1:]) for j in range(len(pron['phones'])-1) if pron['phones'][j].lower() in arpabet_consonants and pron['phones'][j+1].lower() in arpabet_consonants]
            dels += [s.upper() for s in del_c if ' '.join(strip_x(s.split())).upper() not in prns]
            del_c = [' '.join(pron['phones'][:j+1] + pron['phones'][j+2:]) for j in range(len(pron['phones'])-1) if pron['phones'][j].lower() in arpabet_consonants and pron['phones'][j+1].lower() in arpabet_consonants]
            dels += [s.upper() for s in del_c if ' '.join(strip_x(s.split())).upper() not in prns]
    return [{'phones':s.split()} for s in list(set(dels))]

def get_insertions(prons, l1):
    ins = []
    prns = [' '.join(strip_x(pron['phones'])).upper() for pron in prons]
    for pron in prons:
        if l1 in epenthesis and epenthesis[l1]:
            for v in mono_vowels:
                ins += [' '.join(pron['phones'][:j+1] + ['X'+v] + pron['phones'][j+1:]
                                 ) for j in range(len(pron['phones'])-1) if pron['phones'][j].lower() 
                         in arpabet_consonants and pron['phones'][j+1].lower() in arpabet_consonants and ' '.join(strip_x(pron['phones'][:j+1] + [v] + pron['phones'][j+1:])).upper() not in prns]
        if l1 in final_insertion and final_insertion[l1]:
            if pron['phones'][-1].lower() in arpabet_consonants:
                ins_c = ' '.join(pron['phones'] + ['XAX']).upper()
                if ins_c not in ins and ' '.join((strip_x(pron['phones'])+['XAX'])).upper() not in prns:
                    ins += [ins_c]
    return [{'phones':s.split()} for s in list(set(ins))]

def get_errors(prons, l1):
    return get_substitutions(prons, l1) + get_deletions(prons, l1) + get_insertions(prons, l1)

def get_accent_errors(dct, l1):
    new_dict = {}
    pbar = ProgressBar(widgets=[l1+': ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA()], maxval=len(dct))
    pbar.start()
    j = 0
    for word in dct:
        j += 1
        if j % 100 == 0:
            pbar.update(j)
        new_dict[word] = dct[word].copy()
        errors = get_errors(dct[word], l1)
        if len(errors)+len(new_dict[word]) <= 500:
            new_dict[word] += get_errors(dct[word], l1)
    pbar.finish()
    return new_dict

def get_all_accent_errors(dct, l1):
    e = get_errors_from_params(final_deletion[l1] if l1 in final_deletion else False,
                               epenthesis[l1] if l1 in epenthesis else False,
                               cluster_reduction[l1] if l1 in cluster_reduction else False,
                               final_insertion[l1] if l1 in final_insertion else {},
                               substitutions[l1])
    new_dict = {}
    pbar = ProgressBar(widgets=[l1+': ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA()], maxval=len(dct))
    pbar.start()
    j = 0
    for word in dct:
        j += 1
        if j % 100 == 0:
            pbar.update(j)
        new_dict[word] = dct[word].copy()
        errors = get_errors(dct[word], l1)
        if len(errors)+len(new_dict[word]) <= 500:
            new_dict[word] += get_errors(dct[word], l1)
    pbar.finish()
    return new_dict

def gen_dict_files(dct,lang,n, out_path, position_marker=False):
    curr_dict = dct
    for j in range(n):
        curr_dict = get_accent_errors(curr_dict, lang)
        write_dict(curr_dict, out_path + lang+str(j)+'.dct', j == n -1, position_marker)
    for j in range(n-1):
         write_dict(read_dct(out_path+lang+str(j)+'.dct'), out_path+lang+str(j)+'.dct', True, position_marker)
    
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/accent_error_models.cmds', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

parser = argparse.ArgumentParser('Generates theoretical accent error models')
parser.add_argument('-DICT', default='/home/dawna/tts/tools/usr-hts-Ubuntu-12.04-x86_64/festival/lib/dicts/combilex/combilex-rpx.out', help='Dictionary of cardinal pronunciations (.dct or .out)')
parser.add_argument('-OUTDIR', default='lib/dicts',help='Directory in which to store output dicts')
parser.add_argument('-L1', default='all', help='L1 to generate pronunciation errors for')
parser.add_argument('-N', type=int, default=4, help='Number of accent errors to allow per word')
parser.add_argument('-POS', action='store_true', help='Output dictionary with lower case phones followed by position markers e.g. k^I ae^M t^F')
args = parser.parse_args()
dict_type = args.DICT[-4:]
dict_name = args.DICT.split('/')[-1].split('.')[0]
if dict_type == '.out':
    dct = read_combilex(args.DICT)
    write_dict(dct,args.OUTDIR+'/'+dict_name+'.dct', position_marker=args.POS)
elif dict_type == '.dct':
    dct = read_dct(args.DICT)
else:
    raise ValueError('Input -DICT should end in .dct or .out')
if not os.path.isdir(args.OUTDIR):
    os.makedirs(args.OUTDIR)
l1s = [k for k in substitutions] if args.L1 == 'all' else [args.L1]
for l1 in l1s:
    n = 2 if l1 in ['Chinese','Thai','Vietnamese'] else args.N
    gen_dict_files(dct, l1, n, args.OUTDIR+'/'+dict_name+'+', position_marker=args.POS)
    dict_minus(args.OUTDIR+'/'+dict_name+'+'+l1+str(n-1)+'.dct',args.DICT if dict_type == '.dct' else args.OUTDIR+'/'+dict_name+'.dct', args.OUTDIR+'/'+dict_name+'-'+l1+str(n-1)+'.dct')
    
