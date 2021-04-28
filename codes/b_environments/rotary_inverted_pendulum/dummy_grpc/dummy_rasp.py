import grpc
import dummy_grpc_pb2_grpc
from dummy_grpc_pb2 import Response
from concurrent import futures

class DummyGrpc:
    def test(self, request, context):
        return Response(
            message='OK',
            arm_angle=100, arm_velocity=100,
            link_1_angle=100, link_1_velocity=100,
            link_2_angle=100, link_2_velocity=100
        )

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
    dummy_grpc_pb2_grpc.add_DUMMYServicer_to_server(DummyGrpc(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()