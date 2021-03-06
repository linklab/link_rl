# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import quanser_service_pb2 as quanser__service__pb2

class QuanserRIPStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.reset = channel.unary_unary(
                '/QuanserRIP/reset',
                request_serializer=quanser__service__pb2.QuanserRequest.SerializeToString,
                response_deserializer=quanser__service__pb2.QuanserResponse.FromString,
                )
        self.step = channel.unary_unary(
                '/QuanserRIP/step',
                request_serializer=quanser__service__pb2.QuanserRequest.SerializeToString,
                response_deserializer=quanser__service__pb2.QuanserResponse.FromString,
                )
        self.pendulum_reset = channel.unary_unary(
                '/QuanserRIP/pendulum_reset',
                request_serializer=quanser__service__pb2.QuanserRequest.SerializeToString,
                response_deserializer=quanser__service__pb2.QuanserResponse.FromString,
                )
        self.step_sync = channel.unary_unary(
                '/QuanserRIP/step_sync',
                request_serializer=quanser__service__pb2.QuanserRequest.SerializeToString,
                response_deserializer=quanser__service__pb2.QuanserResponse.FromString,
                )
        self.reset_sync = channel.unary_unary(
                '/QuanserRIP/reset_sync',
                request_serializer=quanser__service__pb2.QuanserRequest.SerializeToString,
                response_deserializer=quanser__service__pb2.QuanserResponse.FromString,
                )


class QuanserRIPServicer(object):
    """Missing associated documentation comment in .proto file."""

    def reset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def step(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def pendulum_reset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def step_sync(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def reset_sync(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_QuanserRIPServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'reset': grpc.unary_unary_rpc_method_handler(
                    servicer.reset,
                    request_deserializer=quanser__service__pb2.QuanserRequest.FromString,
                    response_serializer=quanser__service__pb2.QuanserResponse.SerializeToString,
            ),
            'step': grpc.unary_unary_rpc_method_handler(
                    servicer.step,
                    request_deserializer=quanser__service__pb2.QuanserRequest.FromString,
                    response_serializer=quanser__service__pb2.QuanserResponse.SerializeToString,
            ),
            'pendulum_reset': grpc.unary_unary_rpc_method_handler(
                    servicer.pendulum_reset,
                    request_deserializer=quanser__service__pb2.QuanserRequest.FromString,
                    response_serializer=quanser__service__pb2.QuanserResponse.SerializeToString,
            ),
            'step_sync': grpc.unary_unary_rpc_method_handler(
                    servicer.step_sync,
                    request_deserializer=quanser__service__pb2.QuanserRequest.FromString,
                    response_serializer=quanser__service__pb2.QuanserResponse.SerializeToString,
            ),
            'reset_sync': grpc.unary_unary_rpc_method_handler(
                    servicer.reset_sync,
                    request_deserializer=quanser__service__pb2.QuanserRequest.FromString,
                    response_serializer=quanser__service__pb2.QuanserResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'QuanserRIP', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class QuanserRIP(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def reset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/QuanserRIP/reset',
            quanser__service__pb2.QuanserRequest.SerializeToString,
            quanser__service__pb2.QuanserResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def step(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/QuanserRIP/step',
            quanser__service__pb2.QuanserRequest.SerializeToString,
            quanser__service__pb2.QuanserResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def pendulum_reset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/QuanserRIP/pendulum_reset',
            quanser__service__pb2.QuanserRequest.SerializeToString,
            quanser__service__pb2.QuanserResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def step_sync(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/QuanserRIP/step_sync',
            quanser__service__pb2.QuanserRequest.SerializeToString,
            quanser__service__pb2.QuanserResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def reset_sync(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/QuanserRIP/reset_sync',
            quanser__service__pb2.QuanserRequest.SerializeToString,
            quanser__service__pb2.QuanserResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
