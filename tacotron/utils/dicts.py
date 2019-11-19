from progressbar import *
from arpabet import *
import itertools

final_deletion = {'Vietnamese': True, 'Chinese':True}#, 'All':True}
epenthesis = {'Chinese':True, 'Vietnamese': True, 'Thai': True}#, 'All':True}
cluster_reduction = {'Chinese':True, 'Vietnamese': True, 'Thai': True}#, 'All':True}
final_insertion = {'Chinese':True}#,'All':True}

substitutions = {
    'Chinese': [['AX','AA'], ['AX','AH'],['AX','OH'], ['R','L'],['R','W'],['TH','DH'],['DH','TH'],
                ['V','B'],['V','W'],['L','N'],['IH','IY'],['IY','IH'],['EH','AE'],['AE','EH'],
                ['UH','UW'],['UW','UH'],['EY','EH'],['EY','AE'],['EH','EY'],['OW','OH'],['OH','OW']],
    'Thai': [['AE','AA'], ['EY','EH'], ['AY','EY'], ['EY','AY'], ['OW','AA'], ['OW','OH'], ['OW','AW'], ['D','T'], ['G','K'], ['F','P'], ['V','P'], ['V','W'], ['TH','S'], ['TH','DH'], ['TH','T'], ['TH','D'], ['DH','D'], ['S','T'], ['Z','S'], ['SH','T'], ['SH','CH'],  ['L','N'], ['N','L'], ['CH','T'], ['R','L'], ['JH','T'], ['JH','SH'],],
    'Vietnamese': [['AE','AA'], ['EY','EH'],['AY','EY'],['EY','AY'], ['OW','AA'], ['OW','OH'], ['OW','AW'], ['D','T'], ['G','K'], ['F','P'], ['V','P'], ['V','W'], ['TH','S'], ['TH','DH'], ['TH','T'], ['TH','D'], ['DH','D'], ['S','T'], ['Z','S'], ['SH','T'], ['SH','CH'], ['L','N'], ['N','L'], ['CH','T'], ['R','L'], ['JH','T'], ['JH','SH'],],    'Arabic': [['AE','AH'], ['UW','AH'], ['IH','EH'], ['ER','IY'], ['ER','AE'], ['EA','EY'], ['UA','UW'], ['IA','IY'], ['AW','AO'], ['P','B'], ['ZH','JH'], ['ZH','SH'], ['ZH','S'], ['ZH','Z'], ['ZH','G'], ['CH','SH'], ['D','T'], ['V','F'],],
    'French': [['H',''], ['HH',''], ['AH','OH'], ['AA','OH'], ['IY','UW'], ['TH','S'], ['TH','Z'], ['DH','S'], ['DH','Z'], ['DH','D'], ['IH','IY'], ['AA','AE'], ['OW','OH'], ['EY','EH'], ['JH','ZH'], ['CH','SH'], ['Y','JH'],],
    'Dutch': [['IH','IY'], ['EH','AE'], ['AE','EH'], ['AH','AE'], ['AH','AX'], ['AX','AH'], ['UH','UW'], ['OW','AA'], ['OW','OH'], ['EY','AY'], ['TH','T'], ['DH','D'], ['TH','S'], ['DH','Z'], ['V','F'], ['F','V'], ['Z','S'], ['S','Z'], ['SH','S'], ['S','SH'], ['W','V'], ['V','W'], ['D','T'], ['T','D'], ['ZH','SH'],],
    'German': [['EH','AE'], ['AE','EH'], ['AO','AA'], ['OW','OH'], ['EY','EH'], ['EY','AY'], ['B','P'], ['TH','D'], ['DH','D'], ['TH','S'], ['DH','Z'], ['V','F'], ['F','V'], ['Z','S'], ['S','Z'], ['SH','CH'], ['CH','HH'], ['W','V'], ['V','W'], ['D','T'], ['T','D'], ['ZH','JH'], ['ZH','SH'], ['JH','SH'], ['JH','Y'], ['Y','JH'],],
    'Polish': [['IY','IH'], ['AE','EH'], ['AA','OW'], ['OH','OW'], ['ER','EH'], ['P','B'], ['P','F'], ['B','P'], ['T','D'], ['D','T'], ['TH',' S'], ['DH','D'], ['L','R'], ['W','V'],],
    'Russian': [['IY','IH'], ['AE','EH'], ['AA','OW'], ['OH','OW'], ['ER','EH'], ['P','B'], ['P','F'], ['B','P'], ['T','D'], ['D','T'], ['TH',' S'], ['DH','D'], ['L','R'], ['W','V'],],
    'Spanish': [['IH','IY'], ['AE','AH'], ['UH','UW'], ['AO','OW'], ['ER','EH'], ['OW','AA'], ['B','V'], ['V','B'], ['DH','TH'], ['D','DH'], ['K','G'], ['K','W'], ['S','Z'], ['S','SH'], ['SH','S'], ['SH','CH'], ['Y','JH'], ['W','B'], ['JH','SH']]
}

#all_dict = {}
#for l1 in substitutions:
#    for pair in substitutions[l1]:
#        all_dict[pair[0]+' '+pair[1]] = None
#substitutions['All'] = [k.split() for k in all_dict]

vowels = ["i","I","e","E","{","y","Y","2","9","1","@","6","3","a","}","8","&","M","7","V","A","u","U","o","O","Q",
         "eI", "aI", "aIr", "aI", "OI", "OIr", "@U", "@r", "@@r", "Or", "aU", "I@", "E@", "U@", "@@","o~","e~"]

vowels_arp = ["i","I","e","E","{","y","Y","2","9","1","@","6","3","a","}","8","&","M","7","V","A","u","U","o","O","Q",
         "eI", "aI", "aIr", "aI", "OI", "OIr", "@U", "@r", "@@r", "Or", "aU", "I@", "E@", "U@", "@@","o~","e~"]

consonants = ["p","b","t","t^","d","ts","dz","tS","dZ","c","J","k","g","q","B","f","v","T","D","s","Z","j","x","G","X",
              "S","?","h","m","F","n","N","l","lw","L","5","4","r","R","P","w","H","z"]

syllabic_consonants = ["m!","n!","l!"]

lang_codes = {'pol': 'Polish',
              'eng': 'English',
              'ibo': 'Igbo',
              'ger': 'German',
              'ara': 'Arabic',
              'hun': 'Hungarian',
              'rus': 'Russian',
              'chi': 'Chinese',
              'tha': 'Thai',
              'kor': 'Korean',
              'fre': 'French',
              'per': 'Persian',
              'spa': 'Spanish',
              'ibb': 'Ibibio',
              'efi': 'Efik',
              'yor': 'Yoruba',
              'edo': 'Edo'}

def distance(a,b):
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if a[-1] == b[-1]:
        cost = 0
    else:
        cost = 1
       
    res = min([distance(a[:-1], b)+1,
               distance(a, b[:-1])+1, 
               distance(a[:-1], b[:-1]) + cost])
    return res

def merge_dicts(dict1, dict2):
    dict_out = {}
    for word in dict1:
        dict_out[word] = dict1[word]
        for pron in dict2[word]:
            if pron not in dict1[word]:
                dict_out[word] += [pron]
    return dict_out

def parse_c(line, out_dict, rp_adjust=False):
    word_split = line.replace('\n','').split('"')
    if len(word_split) != 3:
        raise ValueError('Failed to parse ' + line)
    word = word_split[1]
    ph_split = word_split[2].split('(((')
    if ph_split[1][-1] != ')' or ph_split[1][-2] != ')':
        raise ValueError('Failed to parse ' + ph_split[1])
    syll_seq, stress_seq = zip(*[s.split(')') for s in ph_split[1][:-3].split(') ((')])
    syll_seq = [s.split() for s in syll_seq]
    def get_stress(syll, stype):
        if int(stype) == 0:
            return [0]*len(syll)
        na = [p for p in syll if p not in vowels and p not in consonants and p not in syllabic_consonants]
        if len(na) > 0:
            raise ValueError('Phone(s) not found: ', na, word)
        vs = [p for p in syll if p in vowels or p in syllabic_consonants]
        if len(vs) == 0:
            raise ValueError('No vowel in ', syll, word, stress_seq, syll_seq)
        if len(vs) > 2:
            raise ValueError('More than one vowel in ', syll)
        vowel = vs[0]
        return [int(stype) if syll[j] == vowel else 0 for j in range(len(syll))]
        
    stress_seq = list(itertools.chain.from_iterable([get_stress(syll_seq[s],
                                                                stress_seq[s]) for s in range(len(stress_seq))]))
    syll_seq = list(itertools.chain.from_iterable(syll_seq))
    if word not in out_dict:
        out_dict[word] = []
    entry = {'phones': [sampa2arpabet(s) for s in syll_seq], 'primary_stress': stress_seq.index(1) if 1 in stress_seq else None,
             'secondary_stress': stress_seq.index(2) if 2 in stress_seq else None}

    if entry not in out_dict[word]:
        out_dict[word] += [entry]
        if rp_adjust and entry['phones'][-1].upper() == 'R':
            entry2 = entry.copy()
            entry2['phones'] = entry['phones'][:-1]
            if entry2 not in out_dict[word]:
                out_dict[word] += [entry2]
        
        
def read_combilex(path, rp_adjust=False):
    out_dict = {}
    with open(path,'r') as fin:
        fin.readline()
        for line in fin.readlines():
            try:
                parse_c(line, out_dict, rp_adjust)
            except ValueError as e:
                print(e)
    return out_dict
        
def read_dct(path):
    out_dict = {}
    with open(path,'r') as fin:
        fin.readline()
        for line in fin.readlines():
            try:                
                word = line.split()[0]
                phones = [p.split('^')[0].upper() for p in line.split()[1:]]
                if word not in out_dict:
                    out_dict[word] = []
                entry = {'phones': phones, 'primary_stress': None, 'secondary_stress': None}
                if entry not in out_dict[word]:
                    out_dict[word] += [entry]
            except ValueError as e:
                print(e)
    return out_dict

def sort_uniq(fname):
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    with open(fname,'w') as fout:
        fout.write(''.join(sorted(list(set(lines)))))

def get_phone(ph, j, last, position_marker):
    pos = 'I' if j == 0 else ('F' if j == -1 else 'M' )
    phone = ph.lower()[1:] if (ph.lower()[0] == 'x' and last) else ph
    return (phone.lower()+'^'+pos) if position_marker else phone.upper()

def write_dict(in_dict, out_file, last=False, position_marker=False):
    with open(out_file,'w') as fout:
        lines = ['\n'.join([' '.join([word.upper()] + [
                            get_phone(pron['phones'][j], -1 if j == len(pron['phones']) - 1 else j, last, position_marker) for j in range(len(pron['phones']))
                ]) for pron in in_dict[word]]) for word in in_dict]
        for line in lines:
            fout.write(line.replace('_%PARTIAL%','_%partial%')+'\n')
    sort_uniq(out_file)

def strip_dicts(words_r, words_g, name):
    pbar = ProgressBar(widgets=['Dicts: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
               ' ', ETA()], maxval=len(os.listdir('dicts')))
    pbar.start()
    j = 0
    for dr in os.listdir('dicts'):
        j += 1
        with open('dicts/'+dr, 'r') as fin:
            out_lines = ''.join([l for l in fin.readlines() if l.split()[0].lower() in (words_r if 'RP' in dr else words_g)])
        with open('dicts_'+name+'/'+dr, 'w') as fout:
            fout.write(out_lines)
        pbar.update(j)
    pbar.finish()

def read_metadata(path):
    with open(path,'r') as fin:
        return {line.split()[1]: line.lower().split()[3] for line in fin.readlines()[1:]}
    
def parse_tsv(line, meta_data, header):
    correct = line.split('\t')[header.split().index('"pronunciationCorrection"')].replace('"','').replace('ER','AX R').split()
    word = line.split('\t')[header.split().index('"token"')]
    file_id = line.split('\t')[header.split().index('"file_id"')].split('_')[0]
    l1 = meta_data[file_id]
    actual = line.split('\t')[header.split().index('"pronunciationTranscript"')].replace('"','').replace('ER','AX R').split()
    return word,actual,correct,l1


def dict_from_tsv(tsv_folder, metadata_file, rp_dict, gm_dict):
    meta_data = read_metadata(metadata_file)
    dct = []
    for path in os.listdir(tsv_folder):
        if '.tsv' in path:
            with open(tsv_folder+'/'+path,'r') as fin:
                header = fin.readline() 
                lines = fin.readlines()
                dct += [parse_tsv(line, meta_data, header) for line in lines if line.split('\t')[header.split().index('"pronunciationCorrection"')] != '"n/a"']
    print('Number of errorful words:', len(dct), 'unique:', len(list(set([w[0] for w in dct]))))
    dct_r = [w for w in dct if w[0].upper() in rp_dict]
    dct_g = [w for w in dct if w[0].upper() in gm_dict]
    print('Of which word in combilex RP:', len(dct_r), 'unique:', len(list(set([w[0] for w in dct_r]))))
    print('Of which word in combilex GAM:', len(dct_g), 'unique:', len(list(set([w[0] for w in dct_g]))))
    return dct_r, dct_g, dct

def get_corr(entry, incorrect, j_array, pbar):
    j_array[0] += 1
    pbar.update(j_array[0])
    if len(entry) == 1:
        return entry[0]['phones']
    distances = {}
    min_d = 1000
    argmin = 0

    for k in range(len(entry)):
        if len(incorrect) > 10:
            return entry[0]['phones']
        d = distance(entry[k]['phones'], incorrect)
        if d < min_d:
            min_d = d
            argmin = k
            if d == 1:
                break
    return entry[argmin]['phones']
    
def get_l1(f, all_files):
    typ = f.split('-')[0].split('CLP')[1].replace('INT','free')
    code = f.split('-')[1][:2].lower()+'_'
    level = f.split('-')[1][2:4].lower()
    return lang_codes[[a for a in all_files if typ in a and code in a and level in a and 'eng' in a][0].split('_')[1]]
    
def dict_from_mlf(file_list, wmlf, mmlf, rp_dict, gm_dict):
    with open(file_list,'r') as fin:
        all_files = fin.readlines()
    with open(wmlf,'r') as fin:
        lines = fin.readlines()
        files = ''.join(lines[1:]).split('.lab"')
        l1s = [get_l1(f.split('\n')[-1], all_files) for f in files[:-1]]
        word_files = [[p.split()[2] for p in s.split('\n')[1:-2]] for s in files[1:]]
    with open(mmlf,'r') as fin:
        lines = fin.readlines()
        files = ''.join(lines[1:]).split('.lab"')
        pron_files = [[[p.split()[2] for p in w.split('\n')[1:-1]] for w in s.split('sil')[1:-1]] for s in files[1:]]

    words = []
    prons = []
    correct_prons = []
    langs = []

    for j in range(len(word_files)):
        if len(word_files[j]) == len(pron_files[j]):
            words += word_files[j]
            prons += pron_files[j]
            langs += [l1s[j]]*len(word_files[j])
    dct = [w for w in list(zip(*(words,prons,langs))) if w[2] in substitutions]

    print('Number of errorful words:', len(dct), 'unique:', len(list(set([w[0] for w in dct]))))
    
    j_array = [0]
    
    pbar = ProgressBar(widgets=['Words: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
               ' ', ETA()], maxval=len(dct))
    pbar.start()
    dct_r = [(w[0], w[1], get_corr(rp_dict[w[0].upper()], w[1], j_array, pbar), w[2]) for w in dct if w[0].upper() in rp_dict and ' '.join(w[1]).upper() not in {' '.join(p['phones']).upper():True for p in rp_dict[w[0].upper()]}]
    pbar.finish()
    j_array = [0]
    pbar = ProgressBar(widgets=['Words: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
               ' ', ETA()], maxval=len(dct))
    pbar.start()
    dct_g = [(w[0], w[1], get_corr(gm_dict[w[0].upper()], w[1], j_array, pbar), w[2]) for w in dct if w[0].upper() in gm_dict and ' '.join(w[1]).upper() not in {' '.join(p['phones']).upper():True for p in gm_dict[w[0].upper()]}]
    pbar.finish()
    print('Of which word in combilex RP:', len(dct_r), 'unique:', len(list(set([w[0] for w in dct_r]))))
    print('Of which word in combilex GAM:', len(dct_g), 'unique:', len(list(set([w[0] for w in dct_g]))))
    return dct_r, dct_g, dct

def read_accent_error_dicts(N, name):
    accent_errors = {}
    for l1 in substitutions:
        accent_errors[l1.lower()] = {}
        for j in range(N):
            path_rp = 'dicts_'+name+'/combilex_RP+'+l1+str(j)+'.dct'
            path_gam = 'dicts_'+name+'/combilex_GAM+'+l1+str(j)+'.dct'
            if os.path.exists(path_rp):
                accent_errors[l1.lower()][j] = (read_dct(path_rp), read_dct(path_gam))
    return accent_errors

def add_ph_marks(l):
    if '^F' not in l:
        l_split = l.split()
        if len(l_split) == 2:
            return l+'^F'
        if len(l_split) == 3:
            return ' '.join([l_split[0],l_split[1]+'^I',l_split[2]+'^F'])
        l_split_marks = [l_split[0],l_split[1]+'^I'] + [p+'^M' for p in l_split[2:-1]] + [l_split[-1]+'^F']
        return ' '.join(l_split_marks) + '\n'
    return l if '\n' in l else (l+'\n')

def readlines(f):
    with open(f,'r') as fin:
        return fin.readlines()
    
def writelines(f, lines):
    with open(f,'w') as fout:
        fout.write(''.join(lines).replace('_%PARTIAL%','_%partial%'))

def readdictlines(f):
    lines = readlines(f)
    return {add_ph_marks(f):None for f in lines}

def dict_minus(f1,f2,f3):
    #print(f1,f2,f3)
    dct = readdictlines(f2)
    lines = [add_ph_marks(l) for l in readlines(f1)]
    out_lines = [l for l in lines if l not in dct]
    out_words = {w: None for w in list(set([o.split()[0] for o in out_lines]))}
    writelines(f3, out_lines + [l for l in dct if l.split()[0] not in out_words])
