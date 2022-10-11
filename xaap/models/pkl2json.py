import pickle
import json
import sys
import os

def main():
    is_error =  False

    if len(sys.argv)==1:
        is_error = True
        
    else:
        # open pickle file
        with open(sys.argv[1], 'rb') as infile:
            obj = pickle.load(infile)

        # convert pickle object to json object
        json_obj = json.loads(json.dumps(obj, default=str))

        # write the json file
        with open(
                os.path.splitext(sys.argv[1])[0] + '.json',
                'w',
                encoding='utf-8'
            ) as outfile:
            json.dump(json_obj, outfile, ensure_ascii=False, indent=4)

        # How to run?
        # python ./models/pkl2json.py ./data/models/chiles_rf_20220902115108.pkl

    if is_error:
        print(f'Usage: python {sys.argv[0]} pickle_file.pkl ')

if __name__ == "__main__": 
    main()