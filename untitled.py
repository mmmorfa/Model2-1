import sqlite3
import json
import numpy as np

# Sample data
list_of_dicts = []
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
independent_dict = {'key1': 'value1', 'key2': 'value2'}

# Serialize data
serialized_list_of_dicts = json.dumps(list_of_dicts)
serialized_numpy_array = numpy_array.tobytes()
serialized_independent_dict = json.dumps(independent_dict)

# Connect to the SQLite database
conn = sqlite3.connect('shared.db')
cursor = conn.cursor()

# Create a table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS shared_data (
                    list_of_dicts TEXT,
                    numpy_array BLOB,
                    independent_dict TEXT
                )''')

# Insert data into the database
cursor.execute('''INSERT INTO shared_data (list_of_dicts, numpy_array, independent_dict) VALUES (?, ?, ?)''',
               (serialized_list_of_dicts, serialized_numpy_array, serialized_independent_dict))

# Query the database to retrieve stored data
cursor.execute('''SELECT list_of_dicts, numpy_array, independent_dict FROM shared_data''')
row = cursor.fetchone()

# Deserialize the retrieved data
if row:
    serialized_list_of_dicts, serialized_numpy_array, serialized_independent_dict = row

    # Deserialize list of dictionaries
    list_of_dicts = json.loads(serialized_list_of_dicts)

    # Deserialize NumPy array
    numpy_array = np.frombuffer(serialized_numpy_array, dtype=np.int64).reshape((2, 3))

    # Deserialize independent dictionary
    independent_dict = json.loads(serialized_independent_dict)

    print("List of dictionaries:", list_of_dicts)
    print("NumPy array:", numpy_array)
    print("Independent dictionary:", independent_dict)

list_of_dicts = [{'a': 1, 'b': 2}, {'c': 5, 'd': 4}]
serialized_list_of_dicts = json.dumps(list_of_dicts)

cursor.execute('''UPDATE shared_data SET list_of_{} = ? WHERE rowid = 1'''.format('dicts'), (serialized_list_of_dicts,))

# Commit changes and close connection
conn.commit()
conn.close()