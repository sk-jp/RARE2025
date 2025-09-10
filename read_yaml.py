from addict import Dict
import yaml


def check_value(data, depth=0):
    for key, value in data.items():
#        print("key:", key)
#        print("value:", value)
        if value == "none" or value == "None":
#            print("key:", key, "--> None")
            data[key] = None
        elif isinstance(value, dict):
            check_value(value, depth + 1)
        elif isinstance(value, list):
            for val in value:
                if isinstance(val, dict):
                    check_value(val, depth + 1)
    return data

def read_yaml(fpath='./model.yaml'):
    with open(fpath, mode='r') as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        yml = check_value(yml)
            
        return Dict(yml)


if __name__ == '__main__':
    d = read_yaml('./unet_base.yaml')
    
    for key in d['Model'].keys():
        print(d['Model'][key])

