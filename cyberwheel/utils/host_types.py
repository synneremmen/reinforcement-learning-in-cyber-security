from enum import IntEnum

class HostTypes(IntEnum):
    UNKNOWN = 0
    USER = 1
    MAIL_SERVER = 2
    FILE_SERVER = 3
    WEB_SERVER = 4
    SSH_JUMP_SERVER = 5
    PROXY_SERVER = 6
    DECOY_USER = 7
    DECOY_SERVER = 8

host_types_map = {
    "UNKNOWN": HostTypes.UNKNOWN,
    "USER": HostTypes.USER,
    "MAIL_SERVER": HostTypes.MAIL_SERVER,
    "FILE_SERVER": HostTypes.FILE_SERVER,
    "WEB_SERVER": HostTypes.WEB_SERVER,
    "SSH_JUMP_SERVER": HostTypes.SSH_JUMP_SERVER,
    "PROXY_SERVER": HostTypes.PROXY_SERVER,
    "DECOY_USER": HostTypes.DECOY_USER,
    "DECOY_SERVER": HostTypes.DECOY_SERVER
}