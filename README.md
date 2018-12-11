# comp550-term-project

Author attribution!

Usage instructions:

cd ./source/
python3 ./classifier.py skip_gram 
python3 ./classifier.py func_word 
python3 ./classifier.py func_word skip_gram 

this script will use the feature set specified by the command-line argument to attempt to classify the authors for segmented chunks of text in the ./data/ folder.

Results will be saved in the ./results/ folder, with a time-stamped filename. 

To be implemented: combine multiple features via the commmand line.
