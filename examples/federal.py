import os
from datetime import datetime

import pandas as pd

from jpt.base.utils import arfftocsv
from jpt.learning.distributions import Numeric, SymbolicType, Bool
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


def main():
    print('loading data...')
    try:
        data = pd.read_csv('../examples/data/federal.csv', sep=';').fillna(value='???')
    except FileNotFoundError:
        print('The file containing this dataset is not in the repository, as it is very large.\n'
              'Download it from here first: https://www.openml.org/data/download/21553061/dataset\n'
              'And then use the convert method to generate a csv file from it.')
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

    print('creating types and variables...')
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

    cmte_id = SymbolicVariable('cmte_id', cmte_id_type)
    amndt_ind = SymbolicVariable('amndt_ind', amndt_ind_type)
    rpt_tp = SymbolicVariable('rpt_tp', rpt_tp_type)
    transaction_pgi = SymbolicVariable('transaction_pgi', transaction_pgi_type)  # missing 28%
    image_num = NumericVariable('image_num', Numeric)
    transaction_tp = SymbolicVariable('transaction_tp', transaction_tp_type)
    entity_tp = SymbolicVariable('entity_tp', entity_tp_type)  # missing
    name = SymbolicVariable('name', name_type)  # missing
    city = SymbolicVariable('city', city_type)  # missing
    state = SymbolicVariable('state', state_type)  # missing
    zip_code = SymbolicVariable('zip_code', zip_code_type)
    employer = SymbolicVariable('employer', employer_type)  # missing 10%
    occupation = SymbolicVariable('occupation', occupation_type)  # missing 5%
    transaction_dt = SymbolicVariable('transaction_dt', transaction_dt_type)  # missing
    transaction_amt = NumericVariable('transaction_amt', Numeric)  # missing
    other_id = SymbolicVariable('other_id', other_id_type)  # missing 98%
    tran_id = SymbolicVariable('tran_id', tran_id_type)  # missing
    file_num = NumericVariable('file_num', Numeric)  # missing
    memo_cd = SymbolicVariable('memo_cd', memo_cd_type)  # missing 97%
    memo_text = SymbolicVariable('memo_text', memo_text_type)  # missing 87%
    sub_id = NumericVariable('sub_id', Numeric)

    # variables = [cmte_id, amndt_ind, rpt_tp, transaction_pgi, image_num, transaction_tp, entity_tp, name, city, state,
    #              zip_code, employer, occupation, transaction_dt, transaction_amt, other_id, tran_id, file_num, memo_cd,
    #              memo_text, sub_id]  # all
    variables = [cmte_id, amndt_ind, rpt_tp, image_num, transaction_tp, entity_tp, name, city, state,
                 zip_code, employer, occupation, transaction_dt, transaction_amt]  # reduced
    data = data[['cmte_id', 'amndt_ind', 'rpt_tp', 'image_num', 'transaction_tp', 'entity_tp', 'name', 'city', 'state',
                 'zip_code', 'employer', 'occupation', 'transaction_dt', 'transaction_amt']]
    data = data.sample(frac=0.1)
    started = datetime.now()
    print('Started learning of %s x %s at %s' % (data.shape[0], data.shape[1], started))
    tree = JPT(variables=variables, min_samples_leaf=20)
    tree.learn(columns=data.values.T)
    # tree.plot(title='Federal Election', directory=os.path.join('/tmp', f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-Federal'))
    print('Learning and plotting took %s. Saving...' % (datetime.now() - started))
    tree.save(os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Federal.json'))


def convert():
    arfftocsv('../examples/data/dataset', '../examples/data/federal.csv')


if __name__ == '__main__':
    main()
