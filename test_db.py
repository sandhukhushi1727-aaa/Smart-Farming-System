
from db import get_db_connection

connection = get_db_connection()

if connection:
    print("Successfully connected to MySQL!")
    cursor = connection.cursor()
    cursor.execute("SELECT DATABASE()")
    print("Connected database:", cursor.fetchone()[0])
    cursor.close()
    connection.close()
else:
    print("Could not connect to MySQL.")