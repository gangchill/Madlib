import psycopg2 as pg
import petl

connection = pg.connect('dbname=Sabbir user=Sabbir')
filename = 'SaveDesign2Conversion.xlsx'
table = petl.io.xlsx.fromxlsx(filename, sheet='orders_match', range_string='A1:E261', row_offset=0, column_offset=0)
print table
petl.todb(table, connection, 'unique_orders_match')
