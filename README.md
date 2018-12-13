# comp550-term-project

Author attribution!

Usage instructions:

cd ./source/
python3 ./classifier.py skip_gram 
python3 ./classifier.py func_word 
python3 ./classifier.py func_word skip_gram (uses both func_word and skip_gram features in parallel for classification)
python3 ./classifier.py long_char_gram --perform_cv (performs 5-fold cross validation)

The available features are skip_gram, func_word, long_char_gram, short_char_gram, lemma_gram, word_gram. Any and all of these can be combined in parallel, however the training takes significantly longer for each additional feature set, particularly with cross validation.

this script will use the feature set specified by the command-line argument to attempt to classify the authors for segmented chunks of text in the ./data/ folder. Results are printed to the terminal.
