import fileinput
import io
import os
import re
import sys
from datetime import datetime

import pandas as pd
import arff
import requests

import dnutils
from dnutils import out
from jpt.base.utils import arfftocsv
from jpt.learning.distributions import Numeric, SymbolicType, Bool
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable

logger = dnutils.getlogger('/federal', level=dnutils.DEBUG)


def main():
    logger.info('Trying to load dataset from local file...')
    f_arff = '../examples/data/dataset'
    f_csv = '../examples/data/federal.csv'
    if not os.path.exists(f_csv):
        src = 'https://www.openml.org/data/download/21553061/dataset'
        logger.warning(f'The file containing this dataset is not in the repository, as it is very large.\nI will try downloading file {src} now and convert it to csv...')
        dfile = requests.get(src)
        open(f_arff, 'wb').write(dfile.content)
        # this is an ugly workaround, because the downloaded file contains an erroneous type declaration for the memo_text field. Forgive me..
        regex = re.compile(r"@ATTRIBUTE memo_text {.*}$", re.IGNORECASE)
        with open(f_arff, 'r+') as f:
            data = f.read()
            f.seek(0)
            f.write(regex.sub("@ATTRIBUTE memo_text STRING\n", data))
            f.truncate()
        arfftocsv(f_arff, f_csv)
        logger.info(f'Success!')
    try:
        data = pd.read_csv('../examples/data/federal.csv', sep=';').fillna(value='???')
        logger.info(f'Success! Loaded dataset containing {data.shape[0]} instances of {data.shape[1]} features each')
    except pd.errors.ParserError:
        logger.error('Could not download and/or parse file. Please download it manually and try again.')
        sys.exit(-1)

    # INFO:
    # cmte_id A 9-character alpha-numeric code assigned to a committee by the Federal Election Commission
    # amndt_ind Amendment indicator: Indicates if the report being filed is new (N), an amendment
    # (A) to a previous report or a termination (T) report
    # rpt_tp Indicates the type of report filed, listed here: https://www.fec.gov/campaign-finance-data/
    # report-type-code-descriptions/
    # transaction_pgi This code indicates the election for which the contribution was made. EYYYY
    # (election Primary, General, Other plus election year)
    # transaction_tp Transaction types, listed here: https://www.fec.gov/campaign-finance-data/
    # transaction-type-code-descriptions/
    # entity_tp Entity Type:
    # CAN = Candidate
    # CCM = Candidate Committee
    # COM = Committee
    # IND = Individual (a person)
    # ORG = Organization (not a committee and not a person)
    # PAC = Political Action Committee
    # PTY = Party Organization
    # name Contributor/lender/transfer Name
    # city City
    # state State
    # zip_code ZIP Code
    # transaction_dt Transaction date (YYYY-MM-DD)
    # transaction_amt Transaction Amount
    # other_id For contributions from individuals this column is null. For contributions from candidates
    # or other committees this column will contain that contributor’s FEC ID
    # cand_id A 9-character alpha-numeric code assigned to a candidate by the FEC, which remains the
    # same across election cycles if running for the same office
    # tran_id Only for Electronic Filings. A unique identifier associated with each itemization or transaction appearing in an FEC electronic file. A transaction ID is unique for a specific committee
    # for a specific report

    logger.info('creating types and variables...')
    cmte_id_type = SymbolicType('cmte_id_type', data['cmte_id'].unique())
    amndt_ind_type = SymbolicType('amndt_ind_type', data['amndt_ind'].unique())
    rpt_tp_type = SymbolicType('rpt_tp_type', data['rpt_tp'].unique())
    transaction_pgi_type = SymbolicType('transaction_pgi_type', data['transaction_pgi'].unique())
    transaction_tp_type = SymbolicType('transaction_tp_type', data['transaction_tp'].unique())
    entity_tp_type = SymbolicType('entity_tp_type', data['entity_tp'].unique())
    name_type = SymbolicType('name_type', data['name'].unique())
    city_type = SymbolicType('city_type', data['city'].unique())
    state_type = SymbolicType('state_type', data['state'].unique())
    zip_code_type = SymbolicType('zip_code_type', data['zip_code'].unique())
    employer_type = SymbolicType('employer_type', data['employer'].unique())
    occupation_type = SymbolicType('occupation_type', data['occupation'].unique())
    transaction_dt_type = SymbolicType('transaction_dt_type', data['transaction_dt'].unique())
    other_id_type = SymbolicType('other_id_type', data['other_id'].unique())
    tran_id_type = SymbolicType('tran_id_type', data['tran_id'].unique())
    memo_cd_type = SymbolicType('memo_cd_type', data['memo_cd'].unique())
    memo_text_type = SymbolicType('memo_text_type', data['memo_text'].unique())

    cmte_id = SymbolicVariable('cmte_id', cmte_id_type)  # 7112 unique values
    amndt_ind = SymbolicVariable('amndt_ind', amndt_ind_type)  # 3 unique values
    rpt_tp = SymbolicVariable('rpt_tp', rpt_tp_type)  # 26 unique values
    transaction_pgi = SymbolicVariable('transaction_pgi', transaction_pgi_type)  # missing 28%; 8 unique values
    image_num = NumericVariable('image_num', Numeric)  # 1879157 unique values
    transaction_tp = SymbolicVariable('transaction_tp', transaction_tp_type)  # 11 unique values
    entity_tp = SymbolicVariable('entity_tp', entity_tp_type)  # 1438 missing; 7 unique values
    name = SymbolicVariable('name', name_type)  # 141 missing; 1534998 unique values
    city = SymbolicVariable('city', city_type)  # 2176 missing; 27709 unique values
    state = SymbolicVariable('state', state_type)  # 8657 missing; 67 unique values
    zip_code = SymbolicVariable('zip_code', zip_code_type)  # 86334 unique values
    employer = SymbolicVariable('employer', employer_type)  # missing 10%; 657449 unique values
    occupation = SymbolicVariable('occupation', occupation_type)  # missing 5%; 145304 unique values
    transaction_dt = SymbolicVariable('transaction_dt', transaction_dt_type)  # 354 missing; 967 unique values
    transaction_amt = NumericVariable('transaction_amt', Numeric)  # 5221 unique values
    other_id = SymbolicVariable('other_id', other_id_type)  # missing 98%; 2543 unique values
    tran_id = SymbolicVariable('tran_id', tran_id_type)  # 327 missing; 2999272 unique values
    file_num = NumericVariable('file_num', Numeric)  # 1 missing; 43597 unique values
    memo_cd = SymbolicVariable('memo_cd', memo_cd_type)  # missing 97%; 2 unique values
    memo_text = SymbolicVariable('memo_text', memo_text_type)  # missing 87%; 16607 unique values
    sub_id = NumericVariable('sub_id', Numeric)  # 51675 unique values

    # variables = [cmte_id, amndt_ind, rpt_tp, transaction_pgi, image_num, transaction_tp, entity_tp, name, city, state,
    #              zip_code, employer, occupation, transaction_dt, transaction_amt, other_id, tran_id, file_num, memo_cd,
    #              memo_text, sub_id]  # all
    variables = [cmte_id, amndt_ind, rpt_tp, transaction_tp, entity_tp, city, state,
                 zip_code, transaction_dt, transaction_amt]  # reduced
    data = data[['cmte_id', 'amndt_ind', 'rpt_tp', 'transaction_tp', 'entity_tp', 'city', 'state',
                 'zip_code', 'transaction_dt', 'transaction_amt']]
    data = data.sample(frac=0.5)
    tree = JPT(variables=variables, min_samples_leaf=data.shape[0]*.01)
    tree.learn(columns=data.values.T)
    tree.plot(title='Federal Election', directory=os.path.join('/tmp', f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-Federal'))
    out(tree)
    tree.save(os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Federal.json'))


def convert():
    arfftocsv('../examples/data/dataset', '../examples/data/federal.csv')


if __name__ == '__main__':
    main()
    # convert()
