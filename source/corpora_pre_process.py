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


if __name__ == "__main__":

    file_tags_dict = lemmatize_input_files()
    dump_data_to_files(file_tags_dict)


