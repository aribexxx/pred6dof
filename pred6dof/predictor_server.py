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
origin_motion_csv_filename = input("Enter the csv filename (without extension): ") 
predicted_motion_csv_filename = str(f"pred_{origin_motion_csv_filename}") 
# record_trace = None  # Initialize flag

# Define host and port
HOST = '127.0.0.1'  # Loopback address for localhost
PORT = 12345        # Arbitrary port number

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def is_complete_json(data):
    """Check if the data contains a complete JSON object."""
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False

try:
    server_socket.bind((HOST, PORT))
    print("Socket bound successfully")

    # Listen for incoming connections,here predictor is the server
    server_socket.listen(1)
    print(f'Server is listening for connections on {HOST}:{PORT}...')

    
       
    while True:
        client_socket, client_address = server_socket.accept()
        print('Connected to {}',client_address)
        client_socket.setblocking(False)
        # testing to send to server
        # client_socket.sendall("hiiiioooo".encode())
        buffer = ""
        record_trace = None
        
        while True:
            try: 
                
                # adding utf-8 make sure that decode as json string
                data = client_socket.recv(1024).decode('utf-8')
                # receive_json_messages(client_socket)
                if not data:
                    break
                print('Received data from client: ',data)
                
                
                # Append the received data to the buffer
                buffer += data

                # Process complete JSON objects
                while buffer:
                    # Find the end of the first complete JSON object
                    end = buffer.find('\n')
                    if end == -1:
                        break  # No complete JSON object found yet
                    
                    # Extract the potential JSON object
                    potential_json = buffer[:end + 1]
                    
                    # Check if it's a complete JSON object
                    if is_complete_json(potential_json):
                        # Remove the processed JSON object from the buffer
                        buffer = buffer[end + 1:]
                        
                        # Parse the JSON object
                        origin_motion_json_obj = json.loads(potential_json)

                # # Split the JSON string by newline characters
                # json_lines = json_str.splitlines()

                # # Parse each JSON string into a Python dictionary
                # json_objects = [json.loads(line) for line in json_lines if line.strip()]
                # for json_obj in json_objects:
                
                        # Access pose orientation and position separately
                        timestamp =  origin_motion_json_obj['timestamp']
                        device_id = origin_motion_json_obj['device_id']
                        orientation = origin_motion_json_obj['motion']['pose']['orientation']
                        position = origin_motion_json_obj['motion']['pose']['position']
                        # print(f"From server: Timestamp: {timestamp}, Orientation: {orientation}, Position: {position}")
                        pose = np.array([position[0], position[1], position[2], orientation[0],orientation[1],orientation[2],orientation[3]])
                        runner.run_with_single_pred_win(measure_motion = pose,w = 100)
                        x_pred_arr = np.squeeze(runner.kf.x_post)
                        print(f"x_post: {x_pred_arr}/n")
                        only_pos_arr = x_pred_arr[::2]
                        print(f"only_pos_arr: {only_pos_arr}/n")
                        position = only_pos_arr.tolist()[0:3]
                        orientation = only_pos_arr.tolist()[3:7]
                        # print(f"predicted array with only 7 number pose: {only_pos_arr.tolist()}") 
                        # Create the data with a custom key "pose"lllll
                        predicted_data = json.dumps({
                            "timestamp":timestamp,
                            "device_id":device_id,
                            "predicted_pose":{
                                    "pose":{
                                        "orientation":orientation,
                                        "position":position
                                        
                                        },
                                    "linear_velocity":[0,0,0],
                                    "angular_velocity":[0,0,0]
                            }
                            
                        })
                        print(f"Sending the json obj {predicted_data}")
                        client_socket.sendall(predicted_data.encode('utf-8'))
                        # Write to CSV if recording is enabled
                        if record_trace is None:
                            record_trace = input("If you want to start recording trace, press Enter. Otherwise, press any other key: ") == ""
                        if record_trace:
                            utils.write_measure_single_to_csv(origin_motion_json_obj, csv_filename=f"./data/milis/{origin_motion_csv_filename}.csv")
                            utils.write_predicted_single_to_csv(json.loads(predicted_data), csv_filename=f"./data/milis/{predicted_motion_csv_filename}.csv")
                
                    else:
                        break  # Wait for more data to complete the JSON object
            except socket.timeout:
                print("Socket timed out, no data received")       
            except BlockingIOError:
                print("No data available, continue to other tasks")
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

