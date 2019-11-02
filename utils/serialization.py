import json
import csv
import collections



def write_to_json(data, json_filename):
    ordered_dict = collections.OrderedDict(sorted(data.items()))

    try:
        with open(json_filename, "w") as write_file:
            json.dump(ordered_dict, write_file, indent=4)
    except IOError:
        print("error when writing to json file")

def write_to_csv(data, csv_filename):
    ordered_dict = collections.OrderedDict(sorted(data.items()))

    try:
        with open(csv_filename, "w") as csv_file:
            writer = write_to_csv().DictWriter(csv, fieldnames = [*ordered_dict])

            writer.writeheader()
            for _ in ordered_dict.item():
                writer.writerow(_)
    except IOError:
        print("error when writing to csv file")