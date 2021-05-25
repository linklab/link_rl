import grpc
import time
import dummy_grpc_pb2_grpc
from dummy_grpc_pb2 import Request

SERVER = '10.0.0.5'

class Dummy():
    def __init__(self):
        self.start_time = 0
        self.last_time = 0

        channel = grpc.insecure_channel('{0}:50051'.format(SERVER))
        self.server_obj = dummy_grpc_pb2_grpc.DUMMYStub(channel)

    def test(self):
        for _ in range(100):
            s = time.time()
            _ = self.server_obj.test(Request(message="OK"))
            e = time.time()
            print("{0:1.5f}".format(e - s))

        # s = time.time()
        # future_responses = []
        # for _ in range(100):
        #     future_response = self.server_obj.test.future(Request(message="OK"))
        #     future_responses.append(future_response)
        # for future_ in future_responses:
        #     future_.result()
        # e = time.time()
        # print("{0:1.5f}".format((e-s) / 100))


if __name__ == "__main__":
    dummy = Dummy()
    dummy.test()