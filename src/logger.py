import inspect
import logging
from functools import wraps
from langchain.callbacks.base import BaseCallbackHandler

LOGGER = logging.getLogger(__name__)

def log_func_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        caller_frame = inspect.stack()[3].frame
        caller_globals = caller_frame.f_globals
        caller_locals = caller_frame.f_locals
        caller_module = caller_globals.get('__name__', '<module>')
        caller_class = caller_locals.get('self', None).__class__.__name__ if 'self' in caller_locals else '<Class>'
        caller_function = caller_frame.f_code.co_name
        caller_string = f"{caller_module}.{caller_class}.{caller_function}"
        LOGGER.debug(f"func={func.__name__} from={caller_string} args={args[1:]} kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper

def log_class_methods(cls):
    # Apply decorator to the methods of cls
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value):
            setattr(cls, attr_name, log_func_call(attr_value))

    visited = set([object])
    visited.add(object)

    # Apply decorator to the methods of the superclasses of cls
    def decorate_bases(bases):
        for base in bases:
            if base in visited:
                continue
            visited.add(base)
            for attr_name, attr_value in base.__dict__.items():
                if callable(attr_value) and attr_name not in cls.__dict__:
                    setattr(cls, attr_name, log_func_call(attr_value))
            decorate_bases(base.__bases__)
    decorate_bases(cls.__bases__)

    return cls


@log_class_methods
class LoggingCallbackHandler(BaseCallbackHandler):
    pass