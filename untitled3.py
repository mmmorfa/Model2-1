import sqlite3
import json
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect('shared.db')
cursor = conn.cursor()

# Query the database to retrieve stored data
cursor.execute('''SELECT list_of_dicts FROM shared_data''')
row = cursor.fetchone()
print(row)

# Deserialize the retrieved data
if row:
    #serialized_list_of_dicts = str(row[0])

    # Deserialize list of dictionaries
    list_of_dicts = json.loads(row[0])

    # Deserialize NumPy array
    #numpy_array = np.frombuffer(serialized_numpy_array, dtype=np.int64).reshape((2, 3))

    # Deserialize independent dictionary
    #independent_dict = json.loads(serialized_independent_dict)

    print("List of dictionaries:", list_of_dicts)
    #print("NumPy array:", numpy_array)
    #print("Independent dictionary:", independent_dict)

# Commit changes and close connection
conn.commit()

# Close connection
conn.close()