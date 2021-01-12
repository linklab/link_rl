import grpc
import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from temp.grpc import helloworld_pb2_grpc, helloworld_pb2


def main():
    channel = grpc.insecure_channel('localhost:50051')
    server_obj = helloworld_pb2_grpc.GreeterStub(channel)

    hello_request = helloworld_pb2.HelloRequest(name='you', address="seoul", phone_number="1111")
    response = server_obj.say_hello(hello_request)
    print("Greeter client received: " + response.reply_message)

    input_number = helloworld_pb2.InputNumbers(num1=1000, num2=2000)
    output_number = server_obj.add(input_number)
    print("Greeter client received: {0}".format(output_number.num))


if __name__ == "__main__":
    main()

