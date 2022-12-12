# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ludwig/profiling/proto/dataset_profile.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from ludwig.profiling.proto import whylogs_messages_pb2 as ludwig_dot_profiling_dot_proto_dot_whylogs__messages__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name="ludwig/profiling/proto/dataset_profile.proto",
    package="dataset_profile",
    syntax="proto3",
    serialized_options=b"Z+github.com/ludwig-ai/ludwig/dataset_profile",
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n,ludwig/profiling/proto/dataset_profile.proto\x12\x0f\x64\x61taset_profile\x1a-ludwig/profiling/proto/whylogs_messages.proto"\xf6\x01\n\x0e\x44\x61tasetProfile\x12\x11\n\ttimestamp\x18\x01 \x01(\x03\x12\x14\n\x0cnum_examples\x18\x02 \x01(\x03\x12\x12\n\nsize_bytes\x18\x03 \x01(\x03\x12N\n\x10\x66\x65\x61ture_profiles\x18\x14 \x03(\x0b\x32\x34.dataset_profile.DatasetProfile.FeatureProfilesEntry\x1aW\n\x14\x46\x65\x61tureProfilesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12.\n\x05value\x18\x02 \x01(\x0b\x32\x1f.dataset_profile.FeatureProfile:\x02\x38\x01"I\n\x0e\x46\x65\x61tureProfile\x12\x37\n\x0fwhylogs_metrics\x18\x01 \x01(\x0b\x32\x1e.dataset_profile.ColumnMessageB-Z+github.com/ludwig-ai/ludwig/dataset_profileb\x06proto3',
    dependencies=[
        ludwig_dot_profiling_dot_proto_dot_whylogs__messages__pb2.DESCRIPTOR,
    ],
)


_DATASETPROFILE_FEATUREPROFILESENTRY = _descriptor.Descriptor(
    name="FeatureProfilesEntry",
    full_name="dataset_profile.DatasetProfile.FeatureProfilesEntry",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="key",
            full_name="dataset_profile.DatasetProfile.FeatureProfilesEntry.key",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="value",
            full_name="dataset_profile.DatasetProfile.FeatureProfilesEntry.value",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=b"8\001",
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=272,
    serialized_end=359,
)

_DATASETPROFILE = _descriptor.Descriptor(
    name="DatasetProfile",
    full_name="dataset_profile.DatasetProfile",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="timestamp",
            full_name="dataset_profile.DatasetProfile.timestamp",
            index=0,
            number=1,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="num_examples",
            full_name="dataset_profile.DatasetProfile.num_examples",
            index=1,
            number=2,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="size_bytes",
            full_name="dataset_profile.DatasetProfile.size_bytes",
            index=2,
            number=3,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="feature_profiles",
            full_name="dataset_profile.DatasetProfile.feature_profiles",
            index=3,
            number=20,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[
        _DATASETPROFILE_FEATUREPROFILESENTRY,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=113,
    serialized_end=359,
)


_FEATUREPROFILE = _descriptor.Descriptor(
    name="FeatureProfile",
    full_name="dataset_profile.FeatureProfile",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="whylogs_metrics",
            full_name="dataset_profile.FeatureProfile.whylogs_metrics",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=361,
    serialized_end=434,
)

_DATASETPROFILE_FEATUREPROFILESENTRY.fields_by_name["value"].message_type = _FEATUREPROFILE
_DATASETPROFILE_FEATUREPROFILESENTRY.containing_type = _DATASETPROFILE
_DATASETPROFILE.fields_by_name["feature_profiles"].message_type = _DATASETPROFILE_FEATUREPROFILESENTRY
_FEATUREPROFILE.fields_by_name[
    "whylogs_metrics"
].message_type = ludwig_dot_profiling_dot_proto_dot_whylogs__messages__pb2._COLUMNMESSAGE
DESCRIPTOR.message_types_by_name["DatasetProfile"] = _DATASETPROFILE
DESCRIPTOR.message_types_by_name["FeatureProfile"] = _FEATUREPROFILE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DatasetProfile = _reflection.GeneratedProtocolMessageType(
    "DatasetProfile",
    (_message.Message,),
    {
        "FeatureProfilesEntry": _reflection.GeneratedProtocolMessageType(
            "FeatureProfilesEntry",
            (_message.Message,),
            {
                "DESCRIPTOR": _DATASETPROFILE_FEATUREPROFILESENTRY,
                "__module__": "ludwig.profiling.proto.dataset_profile_pb2"
                # @@protoc_insertion_point(class_scope:dataset_profile.DatasetProfile.FeatureProfilesEntry)
            },
        ),
        "DESCRIPTOR": _DATASETPROFILE,
        "__module__": "ludwig.profiling.proto.dataset_profile_pb2"
        # @@protoc_insertion_point(class_scope:dataset_profile.DatasetProfile)
    },
)
_sym_db.RegisterMessage(DatasetProfile)
_sym_db.RegisterMessage(DatasetProfile.FeatureProfilesEntry)

FeatureProfile = _reflection.GeneratedProtocolMessageType(
    "FeatureProfile",
    (_message.Message,),
    {
        "DESCRIPTOR": _FEATUREPROFILE,
        "__module__": "ludwig.profiling.proto.dataset_profile_pb2"
        # @@protoc_insertion_point(class_scope:dataset_profile.FeatureProfile)
    },
)
_sym_db.RegisterMessage(FeatureProfile)


DESCRIPTOR._options = None
_DATASETPROFILE_FEATUREPROFILESENTRY._options = None
# @@protoc_insertion_point(module_scope)
