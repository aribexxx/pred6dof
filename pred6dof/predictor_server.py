import pickle
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

runner = predictor_runner.KalmanRunner(pred_window, dataset_path, results_path)
csv_filename = input("Enter the csv filename (without extension): ") 
record_trace = None  # Initialize flag

# Define host and port
HOST = '127.0.0.1'  # Loopback address for localhost
PORT = 12345        # Arbitrary port number

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def receive_json_messages(client_socket):
    buffer = ""
    
    while True:
        try:
            # Receive data from the socket
            data = client_socket.recv(1024).decode()
            
            # If no data is received, the connection is closed
            if not data:
                break

            # Add received data to the buffer
            buffer += data
            
            while True:
                # Attempt to find the boundary of the next JSON message
                try:
                    # Try to parse a JSON message from the buffer
                    message, index = json.JSONDecoder().raw_decode(buffer)
                    
                    # If successful, remove the parsed message from the buffer
                    buffer = buffer[index:].lstrip()
                    
                    # Process the message
                    print("Received JSON message:", message)
                    
                except json.JSONDecodeError:
                    # If the buffer does not contain a complete JSON message, break the loop
                    break

        except ConnectionError as e:
            print(f"Connection error: {e}")
            break
    
    print("Connection closed.")


try:
    server_socket.bind((HOST, PORT))
    print("Socket bound successfully")

    # Listen for incoming connections,here predictor is the server
    server_socket.listen(1)
    print(f'Server is listening for connections on {HOST}:{PORT}...')
    buffer = ""
    
       
    while True:
        client_socket, client_address = server_socket.accept()
        print('Connected to {}',client_address)
   
        # testing to send to server
        # client_socket.sendall("hiiiioooo".encode())
        while True:
            try:
                # adding utf-8 make sure that decode as json string
                json_str = client_socket.recv(1024).decode('utf-8')
                # receive_json_messages(client_socket)
                if not json_str:
                    break
                
                print('Received data from client: ',json_str)
                # Split JSON objects separated by newline
                # Split the JSON string by newline characters
                json_lines = json_str.splitlines()

                # Parse each JSON string into a Python dictionary
                json_objects = [json.loads(line) for line in json_lines if line.strip()]
                for json_obj in json_objects:
                    # Access pose orientation and position separately
                    timestamp =  json_obj['timestamp']
                    orientation = json_obj['motion']['pose']['orientation']
                    position = json_obj['motion']['pose']['position']
                    print(f"Timestamp: {timestamp}, Orientation: {orientation}, Position: {position}")
                    pose = np.array([position[0], position[1], position[2], orientation[0],orientation[1],orientation[2],orientation[3]])
                    runner.run_with_single_pred_win(measure_motion = pose,w = 100)
                    x_pred_arr = np.squeeze(runner.kf.x)
                    only_pos_arr = x_pred_arr[1::2]
                    print(f"predicted array with only 7 number pose: {only_pos_arr.tolist()}") 
                    # Create the data with a custom key "pose"
                    json_data = json.dumps({
                        "timestamp":timestamp,
                        "predicted_pose": only_pos_arr.tolist()
                    })
                    print(f"json data {json_data}")
                    # client_socket.sendall("Hiiiiiii from predictor".encode())
                    #TODO: send predicted result as array back to server.
                    client_socket.send(json_data.encode())
                    # Write to CSV if recording is enabled
                    if record_trace is None:
                        record_trace = input("If you want to start recording trace, press Enter. Otherwise, press any other key: ") == ""
                    if record_trace:
                        utils.write_single_to_csv(json_obj, csv_filename=f"./data/{csv_filename}.csv")
                
            except ConnectionResetError as cre:
                print(f"Connection was reset: {cre}")
                break
            except json.JSONDecodeError as jde: 
                print(f"JSON decoding failed: {jde}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

except KeyboardInterrupt:
    print("Keyboard exit triggered by user.")
    
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

