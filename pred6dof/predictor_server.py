import socket
from .utils import deserialize_to_dictionary
from filterpy.kalman import KalmanFilter

# Define host and port
HOST = '127.0.0.1'  # Loopback address for localhost
PORT = 12345        # Arbitrary port number

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((HOST, PORT))

# Listen for incoming connections
server_socket.listen(1)

print(f'Server is listening for connections on {HOST}:{PORT}...')

while True:
    # Accept incoming connection
    client_socket, client_address = server_socket.accept()
    print('Connected to {client_address}')

    # Receive data from client
    data = client_socket.recv(1024).decode()

    deserialized_data = deserialize_to_dictionary(data)
    print('pos data from client: {}',deserialized_data['linear_velocity'])
    print('orientation from client: {}',deserialized_data['pos'])
    print('Received data from client: {}',deserialized_data.__str__)
    
    # Process data (optional)
    # Here, you can perform any processing on the received data
    
    # Send response back to client
    response = 'Message received!'
    client_socket.sendall(response.encode())

    # Close the connection with the client
    client_socket.close()



