from random import randint
from math import log2, ceil, floor
import sqlite3
import json
import numpy as np

processed_requests = []
PRB_map = np.zeros((14, 10))

def simulate_noise():
    processed_requests = read_parameter_db('processed_requests', 0)

    processed_requests = processed_requests[:2]
    print(processed_requests)

    index_request = randint(0, (len(processed_requests)-1))
    print(index_request)

    processed_requests[index_request]['UE_SiNR'] = randint(1, 20)
    print(processed_requests)

    update_db('processed_requests', 0, processed_requests)

def read_parameter_db(parameter, number):
        # Connect to the SQLite database
        #conn = sqlite3.connect('data/Global_Parameters{}.db'.format(str(self.select_db)))
        conn = sqlite3.connect('data/Global_Parameters401.db')
        cursor = conn.cursor()

        if parameter == 'processed_requests':

            # Query the database to retrieve stored data
            cursor.execute('''SELECT processed_requests FROM Parameters''')
            row = cursor.fetchone()
            return json.loads(row[0])

        if parameter == 'PRB_map':

            # Query the database to retrieve stored data
            cursor.execute('''SELECT PRB_map FROM Parameters''')
            row = cursor.fetchone()
            return np.frombuffer(bytearray(row[0]), dtype=np.int64).reshape((14, 11))

        # Commit changes and close connection
        conn.commit()

        # Close connection
        conn.close()

def update_db(parameter, number, processed_requests):
        # Connect to the SQLite database
        #conn = sqlite3.connect('data/Global_Parameters{}.db'.format(str(self.select_db)))
        conn = sqlite3.connect('data/Global_Parameters4.db')
        cursor = conn.cursor()

        if parameter == 'processed_requests':
            # Serialize data
            serialized_parameter = json.dumps(processed_requests)
            print(serialized_parameter)

            cursor.execute('''UPDATE Parameters SET processed_requests = ? WHERE rowid = 1''', (serialized_parameter,))

        if parameter == 'PRB_map':
            # Serialize data
            serialized_parameter = PRB_map.tobytes()

            cursor.execute('''UPDATE Parameters SET PRB_map = ? WHERE rowid = 1''', (serialized_parameter,)) 

        # Commit changes and close connection
        conn.commit()
        conn.close()


#simulate_noise()
        
prb = read_parameter_db("PRB_map", 0)
print(prb)

req = read_parameter_db("processed_requests", 0)
print(len(req))

a = 0
for i in req:
    if len(i) == 10:
         a+=1

print(a)
#print(req)