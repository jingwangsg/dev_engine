import json
from typing import Sequence
import subprocess
import os
import dill


def load_json(fn):
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, fn):
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def load_jsonl(fn, as_generator=False):
    with open(fn, "r", encoding="utf-8") as f:
        # FIXME: as_generator not working
        # if as_generator:
        #     for line in f:
        #         yield json.loads(line)
        # else:
        return [json.loads(line) for line in f]


def save_jsonl(obj: Sequence[dict], fn):
    with open(fn, "w", encoding="utf-8") as f:
        for line in obj:
            json.dump(line, f)
            f.write("\n")


def load_pickle(fn):
    # return joblib.load(fn)
    with open(fn, "rb") as f:
        obj = dill.load(f)
    return obj


def save_pickle(obj, fn):
    # return joblib.dump(obj, fn, protocol=pickle.HIGHEST_PROTOCOL)
    with open(fn, "wb") as f:
        dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL)


import polars as pl


def load_csv(fn, delimiter=",", has_header=True):
    fr = open(fn, "r")
    encodings = ["ISO-8859-1", "cp1252", "utf-8"]

    # Try using different encoding formats
    def _load_csv_with_encoding(encoding):
        read_csv = pl.read_csv(
            fn,
            separator=delimiter,
            encoding=encoding,
            infer_schema_length=0,
            has_header=has_header,
            n_threads=32,
        )

        if has_header:
            return read_csv.to_dicts()
        else:
            ret_list = []
            dict_list = read_csv.to_dicts()
            num_columns = len(dict_list[0].keys())

            for dict_row in dict_list:
                row_list = []
                for col_idx in range(1, num_columns + 1):
                    column_name = f"column_{col_idx}"
                    row_list += [dict_row[column_name]]
                ret_list += [row_list]

            return ret_list

    for encoding in encodings:
        try:
            df = _load_csv_with_encoding(encoding)
            return df

        except UnicodeDecodeError:
            print(f"Error: {encoding} decoding failed, trying the next encoding format")

    raise ValueError(f"Failed to load csv file {fn}")


def save_csv(obj, fn, delimiter=","):
    df = pl.DataFrame(obj)
    df.write_csv(fn, separator=delimiter, quote_style="non_numeric")