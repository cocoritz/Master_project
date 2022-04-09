#coline Ritz
#Created on 07/03/2022
#this file aims to sniff my network and send it to Mongodb 


import pyshark 
import pymongo
import time
from pymongo import MongoClient

#URI (Uniform Resource Identifier) to connect to MongoDB database 
client_URI = "mongodb+srv://coline:clondonc@cluster0.4olg7.mongodb.net/myFirstDatabase?retryWrites=true&w=majority" 
client = MongoClient(client_URI)
db = client.Tshark # use or create a database named Twitter
test_collection = db.test #use or create a collection named 
test_collection.create_index([("id", pymongo.ASCENDING)],unique = True) # make sure the collected tweets are unique


# define interface - how to capture all interface? 
networkInterface = "en0"

print("listening on %s" % networkInterface)


def capture_live_packets(networkInterface):
    # define capture object
    capture = pyshark.LiveCapture(interface=networkInterface)
    for raw_packet in capture.sniff_continuously():
        print(get_packet_details(raw_packet))

def get_packet_details(packet):
    """
    This function is designed to parse specific details from an individual packet.
    :param packet: raw packet from either a pcap file or via live capture using TShark
    :return: specific packet details
    """
    protocol = packet.transport_layer
    source_address = packet.ip.src
    source_port = packet[packet.transport_layer].srcport
    destination_address = packet.ip.dst
    destination_port = packet[packet.transport_layer].dstport
    packet_time = packet.sniff_time
    packet_length = packet.length

    netpacket = { 
                    'Packet Timestamp': packet_time, 
                    'Protocol type': protocol,
                    'Source address': source_address,
                    'Source port': source_port,
                    'Packet length': packet_length,
                    'Destination address': destination_address,
                    'Destination port': destination_port 
                    }

    print(netpacket)
    
    return netpacket 

while True:

    packet_capture = capture_live_packets('en0')
    
    # Store it!
    net_id = test_collection.insert_one(packet_capture)
    
    
