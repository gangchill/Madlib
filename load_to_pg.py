import psycopg2 as pg
import petl
from file_reader import FileReader

connection = pg.connect('dbname=Sabbir user=Sabbir')

# Get True Reviews
genuine_file = 'TruthfulNew.txt'
true_filereader = FileReader(genuine_file, "True") # new object
true_filereader.parse_file()
genuine_reviews = true_filereader.get_review_list()
print genuine_reviews
true_table = petl.fromdicts(genuine_reviews)
print true_table
petl.todb(true_table, connection, 'true_review')

# Get Fake Reviews
fake_file = 'FakeNew.txt'
fake_file_reader = FileReader(fake_file, "Fake") # new object
fake_file_reader.parse_file()
fake_reviews = fake_file_reader.get_review_list()
print fake_reviews
fake_table = petl.fromdicts(fake_reviews)
print fake_table
petl.todb(fake_table, connection, 'fake_review')
