# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: basic.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0b\x62\x61sic.proto\x12\x05proto\"*\n\tdataframe\x12\x1d\n\x06series\x18\x01 \x03(\x0b\x32\r.proto.series\"\x82\x01\n\x06series\x12\r\n\x05index\x18\x01 \x01(\x03\x12+\n\x07\x63olumns\x18\x02 \x03(\x0b\x32\x1a.proto.series.ColumnsEntry\x1a<\n\x0c\x43olumnsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1b\n\x05value\x18\x02 \x01(\x0b\x32\x0c.proto.value:\x02\x38\x01\"S\n\x05value\x12\x13\n\tint_value\x18\x01 \x01(\x05H\x00\x12\x15\n\x0b\x66loat_value\x18\x02 \x01(\x02H\x00\x12\x16\n\x0cstring_value\x18\x03 \x01(\tH\x00\x42\x06\n\x04\x64\x61tab\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'basic_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_SERIES_COLUMNSENTRY']._options = None
  _globals['_SERIES_COLUMNSENTRY']._serialized_options = b'8\001'
  _globals['_DATAFRAME']._serialized_start=22
  _globals['_DATAFRAME']._serialized_end=64
  _globals['_SERIES']._serialized_start=67
  _globals['_SERIES']._serialized_end=197
  _globals['_SERIES_COLUMNSENTRY']._serialized_start=137
  _globals['_SERIES_COLUMNSENTRY']._serialized_end=197
  _globals['_VALUE']._serialized_start=199
  _globals['_VALUE']._serialized_end=282
# @@protoc_insertion_point(module_scope)
