import os
import sys
from concurrent import futures

import grpc

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from temp.grpc import helloworld_pb2_grpc, helloworld_pb2


class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def say_hello(self, request, context):
        return helloworld_pb2.HelloReply(reply_message='Hello, {0} {1} {2}!'.format(
            request.name, request.address, request.phone_number
        ))

    def add(self, request, context):
        return helloworld_pb2.OutputNumber(num=request.num1 + request.num2)


def serve_main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve_main()