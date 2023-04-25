
import sys
import json
from . import convert_model

def save_model(file_name_base, model):
    """Save and convert Keras model"""
    keras_file = f'{file_name_base}.h5'
    fdeep_file = f'{file_name_base}.json'
    print(f'Saving {keras_file}')
    model.save(keras_file, include_optimizer=False)
    print(f'Converting {keras_file} to {fdeep_file}.')
    convert_model.convert(keras_file, fdeep_file)
    print(f'Conversion of model {keras_file} to {fdeep_file} done.')

def save_features(feature_names, data, types):
    # check that feature_names len == data len and that feature_names len == types len
    feature_names_len = len(feature_names)
    types_len = len(types)
    data_amount = len(data)
    data_len = -1 if data_amount <= 1 else len(data[0])
    print(feature_names_len)
    print(types_len)
    print(data_len)
    if feature_names_len != types_len or types_len != data_len:
        return False # error out if bad format
    print('len fine')
    # check if features names is a 1d list of strings
    for i in feature_names:
        if type(i).__class__ != str.__class__:
            return False # error out if bad format
    print('feature names fine')
    # check if types is a 1d array of float or int type instances
    for i in types: # only support ints and float types atm
        if i != int.__class__ or i != float.__class__:
            return False
    print('types fine')
    # check if data is a 2d array of data
    for i in range(0, ):
        if len(data[i]) != data_len:
            return False
    print('data fine')
    # create list feature_name: (min, max)
    map = list()
    for i in range(0, feature_names_len):
        t = types[i]
        # for each feature_name, find min and max in data
        min, max = 0, 0
        if t == float.__class__:
            min = sys.float_info.max
            max = -sys.float_info.max
        elif t == int.__class__:
            min = sys.maxsize
            max = -sys.maxsize - 1
        for j in data:
            if j[i].item() < min:
                min = j[i].item()
            if j[i].item() > max:
                max = j[i].item()
        map.append({
            "name": feature_names[i],
            "domain_type": "continuous" if types[i] == float.__class__ else "discrete",
            "min": min,
            "max": max
        })
    print(map)
    # create json
    jsonObj = json.dumps(map, indent=4)
    with open("features.json", "w") as out:
        out.write(jsonObj)
    return True
