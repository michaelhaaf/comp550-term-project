import treetaggerwrapper
import glob
import pickle
import os


def lemmatize_input_files():
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')
    files = [f for f in glob.glob("../texts/txt/*.txt")]
    return {os.path.basename(f): treetaggerwrapper.make_tags(tagger.tag_file(f), exclude_nottags=True) for f in files}


def dump_data_to_files(file_data_dict):
    for filename, tags in file_data_dict.items():
        file = open("../data/" + filename, "wb")
        pickle.dump(tags, file)
        file.close()


def load_preprocessed_data():
    print('loading data...', end='')
    file_paths = [f for f in glob.glob("../data/*.txt")]
    input_data_dict = {}
    for path in file_paths:
        file = open(path , "rb")
        input_data_dict[os.path.basename(path)] = pickle.load(file)
        file.close()
        print('.', end='')
    print('done loading data.')
    return input_data_dict

