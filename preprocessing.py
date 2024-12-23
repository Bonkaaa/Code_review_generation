import json
import pandas as pd

def data(path):
    code_tokens = []
    docstring_tokens = []

    #Extract data
    with open(path) as file:
        for line in file:
            try:
                data = json.loads(line)
                code_tokens.append(data['code_tokens'])
                docstring_tokens.append(data['docstring_tokens'])
            except json.decoder.JSONDecodeError:
                print(line)

    #Convert into dataframe
    dicts = {'Code_tokens': code_tokens, 'Docstring_tokens': docstring_tokens}
    df = pd.DataFrame(dicts)
    df_new = df.dropna()

    #Change the data types
    df_new['Code_tokens'] = df_new['Code_tokens'].astype("string")
    df_new['Docstring_tokens'] = df_new['Docstring_tokens'].astype("string")

    return df_new

