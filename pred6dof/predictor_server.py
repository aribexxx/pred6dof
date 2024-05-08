import socket
from filterpy.kalman import KalmanFilter
import json

import numpy as np
import predictor_runner
import utils

"""Runs Kalman filter on all traces and evaluates the results"""
pred_window = 20
dataset_path = "./data/alvr.csv"
results_path = "./results/prediction"

runner = predictor_runner.KalmanRunner(pred_window,
                              dataset_path,
                              results_path)




# Define host and port
HOST = '127.0.0.1'  # Loopback address for localhost
PORT = 12345        # Arbitrary port number

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


try:
    server_socket.bind((HOST, PORT))
    print("Socket bound successfully")

    # Listen for incoming connections
    server_socket.listen(1)
    print(f'Server is listening for connections on {HOST}:{PORT}...')

    while True:
        # Accept incoming connection
        client_socket, client_address = server_socket.accept()
        print('Connected to {}',client_address)

        # TODO:Send prediction result back to client
        response = 'Message received!'
        client_socket.sendall(response.encode())

        # Receive data from client
        json_str = client_socket.recv(1024).decode()
        print('Received data from client: ',json_str)
        # Split JSON objects separated by newline
        json_objs = json_str.strip().split('\n')
        for json_obj in json_objs:
            data_obj = json.loads(json_obj)
            # Access pose orientation and position separately
            timestamp =  data_obj['timestamp']['nanos']
            orientation = data_obj['motion']['pose']['orientation']
            position = data_obj['motion']['pose']['position']
            print(f"Timestamp: {timestamp}, Orientation: {orientation}, Position: {position}")
            pose = np.array([position[0], position[1], position[2], orientation[0],orientation[1],orientation[2],orientation[3]])
            runner.run_with_single_pred_win(motions_array = pose,w = 1)
            #utils.write_single_to_csv(data_obj,filename="./data/alvr.csv")
            #utils.write_sessions_csv_files()


except OSError as e:
    if e.errno == 10048:
        print(f"Port {PORT} is already in use.")
        # Handle this situation, maybe try another port or terminate gracefully
        # If the process is using the specified port, terminate it gracefully

    else:
        # Handle other OSError exceptions
        print("Error:", e)
finally:
    server_socket.close()
