# model-api-frequency
Creates ML models of frequency of API calls

The API integers printed out by api_existence.py are the zero-indexed
line numbers in api.txt

The API integers printed out by attack.py are the line numbers in api.txt
(i.e., equivalent integers to sequence integers)

## Usage
```
# Parse data into CSV file
$ time python parse.py /data/arsa/api-sequences-all-classification-32-filtered/ api.txt data.csv

# Model data & save model to file
$ python api_frequency.py data.csv model.pkl

#TODO
# Attack model
$ python attack.py /data/arsa/api-sequences-all-classification-32-filtered/ api.txt model.pkl /data/arsa/api-frequency-attacks/
```
