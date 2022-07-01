import unittest
import peewee
# raise unittest.SkipTest('Skip the relational test since the connection and content of a database is needed.')

database = peewee.MySQLDatabase('mutagenesis', **{'charset': 'utf8',
                                                  'sql_mode': 'PIPES_AS_CONCAT',
                                                  'use_unicode': True,
                                                  'host': 'relational.fit.cvut.cz',
                                                  'port': 3306,
                                                  'user': 'guest',
                                                  'password': 'relational'})


class UnknownField(object):
    def __init__(self, *_, **__): pass


class BaseModel(peewee.Model):
    class Meta:
        database = database


class Molecule(BaseModel):
    ind1 = peewee.IntegerField()
    inda = peewee.IntegerField()
    logp = peewee.FloatField()
    lumo = peewee.FloatField()
    molecule_id = peewee.CharField(primary_key=True)
    mutagenic = peewee.CharField()

    class Meta:
        table_name = 'molecule'


class Atom(BaseModel):
    atom_id = peewee.CharField(primary_key=True)
    charge = peewee.FloatField()
    element = peewee.CharField()
    molecule = peewee.ForeignKeyField(column_name='molecule_id', field='molecule_id', model=Molecule)
    type = peewee.IntegerField()

    class Meta:
        table_name = 'atom'


class Bond(BaseModel):
    atom1 = peewee.ForeignKeyField(column_name='atom1_id', field='atom_id', model=Atom)
    atom2 = peewee.ForeignKeyField(backref='atom_atom2_set', column_name='atom2_id', field='atom_id', model=Atom)
    type = peewee.IntegerField()

    class Meta:
        table_name = 'bond'
        indexes = (
            (('atom1', 'atom2'), True),
        )
        primary_key = peewee.CompositeKey('atom1', 'atom2')


class RelationalTest(unittest.TestCase):

    def test_db(self):
        """Checks if the database can be connected to."""
        self.assertTrue(database.connect())

    def test_join(self):
        q: peewee.ModelSelect = Molecule.select(Molecule, Atom).join(Atom)
        print([sample.logp for sample in q.execute()], sep="\n")

if __name__ == '__main__':
    unittest.main()
