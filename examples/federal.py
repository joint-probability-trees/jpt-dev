"""Federal Election Commission dataset example.

Downloads the FEC contributions dataset from OpenML,
converts it from ARFF to CSV, and learns a Joint
Probability Tree over a subset of the features.

Dataset fields:
    cmte_id          9-char alpha-numeric committee code
                     assigned by the FEC
    amndt_ind        Amendment indicator (N=new,
                     A=amendment, T=termination)
    rpt_tp           Report type code
    transaction_tp   Transaction type code
    entity_tp        Entity type (CAN, CCM, COM, IND,
                     ORG, PAC, PTY)
    city             Contributor city
    state            Contributor state
    zip_code         Contributor ZIP code
    transaction_dt   Transaction date (YYYY-MM-DD)
    transaction_amt  Transaction amount

Requires the ``arff`` dev dependency for ARFF-to-CSV
conversion::

    pip install pyjpt[dev]
"""
import csv
import logging
import os
import re
import sys
import tempfile
from csv import Dialect
from _csv import register_dialect, QUOTE_NONNUMERIC
from typing import Any, List, Union

import pandas as pd
import requests

from dnutils import ifnone
from jpt.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable

logger: logging.Logger = logging.getLogger(__name__)

# Resolve data paths relative to the script location
# so the example works regardless of the working
# directory.
_SCRIPT_DIR: str = os.path.dirname(
    os.path.abspath(__file__)
)
_DATA_DIR: str = os.path.join(_SCRIPT_DIR, 'data')
_ARFF_PATH: str = os.path.join(_DATA_DIR, 'dataset')
_CSV_PATH: str = os.path.join(_DATA_DIR, 'federal.csv')

_DATASET_URL: str = (
    'https://www.openml.org/data/'
    'download/21553061/dataset'
)

# Features used for training the JPT
_FEATURES: List[str] = [
    'cmte_id',
    'amndt_ind',
    'rpt_tp',
    'transaction_tp',
    'entity_tp',
    'city',
    'state',
    'zip_code',
    'transaction_dt',
    'transaction_amt',
]


# ----------------------------------------------------------------------

class CSVDialect(Dialect):
    """Semicolon-delimited CSV dialect for the FEC
    data."""

    delimiter: str = ';'
    quotechar: str = '"'
    doublequote: bool = True
    skipinitialspace: bool = False
    lineterminator: str = '\r\n'
    quoting: int = QUOTE_NONNUMERIC


register_dialect("csvdialect", CSVDialect)


# ----------------------------------------------------------------------

def _convert_value(
        key: str,
        value: Any
) -> Union[int, float, str]:
    """Convert a single ARFF cell to an appropriate
    Python type.

    Date fields are kept as strings; numeric values are
    cast to int or float if possible, everything else
    becomes a string.

    :param key:   the column/attribute name
    :param value: the raw cell value
    :returns:     the converted value
    """
    value = ifnone(value, '')
    if key == 'transaction_dt':
        return str(value)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return str(value)


def arfftocsv(
        arffpath: str,
        csvpath: str
) -> None:
    """Convert an ARFF file to semicolon-delimited CSV.

    :param arffpath: path to the ARFF input file
    :param csvpath:  path to the CSV output file
    :raises ImportError: if the ``arff`` package is not
                         installed
    """
    try:
        import arff
    except ImportError:
        raise ImportError(
            'The "arff" package is required to convert'
            ' ARFF files. Install it via: '
            'pip install pyjpt[dev]'
        )

    logger.info('Loading arff file: %s', arffpath)
    with open(arffpath, 'r') as f:
        data: dict = arff.load(
            f, encode_nominal=True
        )

    logger.info('Writing to csv file: %s', csvpath)
    fieldnames: List[str] = [
        attr[0] for attr in data.get('attributes')
    ]
    with open(csvpath, 'w', newline='') as csvfile:
        writer: csv.DictWriter = csv.DictWriter(
            csvfile,
            dialect='csvdialect',
            fieldnames=fieldnames
        )
        writer.writeheader()
        for row in data.get('data'):
            writer.writerow(
                {
                    k: _convert_value(k, v)
                    for k, v in zip(fieldnames, row)
                }
            )


# ----------------------------------------------------------------------

def _download_and_convert() -> None:
    """Download the FEC dataset and convert it to CSV.

    The raw ARFF file from OpenML contains an erroneous
    type declaration for the ``memo_text`` field which is
    patched before conversion.

    :raises ImportError: if the ``arff`` package is not
                         installed
    :raises requests.HTTPError: if the download fails
    """
    logger.warning(
        'The dataset is not in the repository '
        '(too large). Downloading from %s ...',
        _DATASET_URL
    )

    # Download the ARFF file
    response: requests.Response = requests.get(
        _DATASET_URL
    )
    response.raise_for_status()
    with open(_ARFF_PATH, 'wb') as f:
        f.write(response.content)

    # Fix erroneous type declaration for memo_text
    regex: re.Pattern = re.compile(
        r"@ATTRIBUTE memo_text {.*}$",
        re.IGNORECASE
    )
    with open(_ARFF_PATH, 'r+') as f:
        content: str = f.read()
        f.seek(0)
        f.write(
            regex.sub(
                "@ATTRIBUTE memo_text STRING\n",
                content
            )
        )
        f.truncate()

    # Convert ARFF to CSV
    arfftocsv(_ARFF_PATH, _CSV_PATH)
    logger.info('Download and conversion successful.')


def _load_data() -> pd.DataFrame:
    """Load the FEC dataset as a pandas DataFrame.

    Downloads and converts the dataset if the CSV file
    does not exist yet.

    :returns: the loaded DataFrame
    """
    if not os.path.exists(_CSV_PATH):
        _download_and_convert()

    logger.info('Loading dataset from %s ...', _CSV_PATH)
    try:
        data: pd.DataFrame = pd.read_csv(
            _CSV_PATH,
            sep=';'
        ).fillna(value='???')
    except pd.errors.ParserError:
        logger.error(
            'Could not parse %s. Please download it '
            'manually and try again.',
            _CSV_PATH
        )
        sys.exit(-1)

    logger.info(
        'Loaded dataset: %d instances, %d features',
        data.shape[0],
        data.shape[1]
    )
    return data


# ----------------------------------------------------------------------

def main(visualize=True) -> None:
    """Train a JPT on the Federal Election dataset.

    :param visualize: whether to show interactive plots
    """
    # Load data and define variables
    data: pd.DataFrame = _load_data()

    symbolic_features: List[str] = [
        f for f in _FEATURES if f != 'transaction_amt'
    ]
    variables: list = [
        SymbolicVariable(
            f,
            SymbolicType(
                f'{f}_type',
                data[f].unique()
            )
        )
        for f in symbolic_features
    ] + [
        NumericVariable('transaction_amt', Numeric)
    ]

    # Subsample and learn tree
    data = data[_FEATURES].sample(frac=0.5)
    tree: JPT = JPT(
        variables=variables,
        min_samples_leaf=int(data.shape[0] * .01)
    )
    tree.learn(data)

    # Plot and save the tree
    out_dir: str = tempfile.mkdtemp(
        prefix='jpt-federal-'
    )
    tree.plot(
        title='Federal Election',
        directory=out_dir,
        view=visualize
    )
    print(tree)
    tree.save(
        os.path.join(out_dir, 'federal.json')
    )


if __name__ == '__main__':
    main(visualize=True)
