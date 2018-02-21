import bluetooth
import time
import sys
import socket

def main():
    # Create socket for bluetooth RFCOMM
    serverSocket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    # Bind host to channel 1
    channel = 1
    serverSocket.bind(("", channel))
    # Accept 1 connection
    serverSocket.listen(1)
    print ("Waiting for connection...")
    #Assign Client socket and MAC Address
    clientSocket, mAddress = serverSocket.accept()
    print ("Connection Accepted from ", mAddress)

    # Receive Data from mobile device (client)
    mobileData = clientSocket.recv(1024)
    serverSocket.close()
    return mobileData
main()