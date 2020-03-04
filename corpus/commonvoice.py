from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset

# Additional (official) text src provided
OFFICIAL_TXT_SRC = ['lm-norm.txt']
# Remove longest N sentence in librispeech-lm-norm.txt
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading LibriSpeech
READ_FILE_THREADS = 2


def read_text(file):
    '''Get transcription of target wave file, 
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread'''
    with open(file, 'r') as fp:
        for line in fp:
            split = line.split('	')
            return split[2], split[1]


class Dataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        global READ_FILE_THREADS
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        
        if READ_FILE_THREADS > len(split):
            READ_FILE_THREADS = len(split)

        # Read text
        text, file_list = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in split)
        #text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
        text = [tokenizer.encode(txt) for txt in text]

        # Sort dataset by text length
        #file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)


class TextDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        read_txt_src = []

        # Read text
        all_sent, _ = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list)
        
        self.text = [tokenizer.encode(txt) for txt in tqdm(all_sent)]
        del all_sent

        # Read file size and sort dataset by file size (Note: feature len. may be different)
        self.text = sorted(self.text, reverse=True, key=lambda x: len(x))

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.text)-self.bucket_size, index)
            # Return a bucket
            return self.text[index:index+self.bucket_size]
        else:
            return self.text[index]

    def __len__(self):
        return len(self.text)
