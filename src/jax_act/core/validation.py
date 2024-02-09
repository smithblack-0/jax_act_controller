import textwrap
from enum import Enum
from typing import Callable, Tuple, List, Any, Type, Optional, Dict
from jax.experimental import checkify
from dataclasses import dataclass
# Define configuration options
#
# Use enums for this.

class _ValidationConfigState:
    validation_detail_level: str = "advanced"
    validation_jit_config: str = "warnings"
    validation_logging_config: Callable[]
_validation_detail_level = "advanced" # One of off, basic, advanced
class ValidationDetailLevel(Enum):
    """
    There are three validation detail
    levels. These are

    off: No validation occurs at all. Fastest
    basic: Only validation that does not require python callbacks is run.
    advanced: All validation is executed
    """
    off: str = "off"
    basic: str = "basic"
    advanced: str = "advanced"

class ValidationJit(Enum):
    """
    Jit has special behavior that has to be
    accounted for when detail level is in
    advanced mode. This specifies it.

    Sometimes, we cannot raise an error immediately
    on encountering it because it would break jax flow
    control. This specifies what to do.

    debug: Raises in a manner which might not be jit compatible
    checkify: Raises using checkify statements. You will need to stage jit yourself
    warnings: Sends incompable errors as messages to the logging location
    """

    debug: str = "debug"
    checkify: str = "checkify"
    warnings: str = "warnings"
class ValidationLogging:
    """
    Because of how the validation API is being
    set up, it intercepts all calls. This makes
    it trivial to log to more than just the console

    console: Raises like normal, and warns like normal
    callback: Captures all warnings and messages, and sends them to a callback
    file: Captures all warnings and messages, and sends them to a specified file
    """
    console: str = "console"
    callback: str = "callback"
    file: str = "file"

# Specify the config object.
class ValidationConfig:
    """
    A data class for controlling how validation is
    executed.

    fields
    ^^^^^^

    There are three fields that must be specified.
    These fields are:

    - :field detail_level: One of "off", "basic", or "advanced
    - :field jit_config: one of "debug", "checkify", or "warnings"
    - :field logging_config: One of "console", "callback", or "file"

    Also, depending on logging config, you might also need to specifiy
    - field callback: The callback to call with errors
    - field file: The file to log to
    """
    detail_level: str = ValidationDetailLevel.advanced
    jit_config: str = ValidationJit.warnings
    logging_config: str = ValidationLogging.console

# When an error occurs, two things are generated.
#
# These are the error type, and the message.
# The emission registry will contain code saying
# how to handle the current configuration

class LoggingRegistry:
    """
    Handles logging operations.

    Logging emissions consist of a string
    describing the error, and the kind of error as a
    type.

    This internal registry needs to accept a name that is
    in validation logging, and
    """
    console_callback: Callable[[str, bool, Type[Exception]], None]
    manual_callback: Callable[[str, bool, Type[Exception]], None]
    file_callback: Callable[[str, Type[Exception]], None]

    def console_callback(self,
                         message: str,
                         jit_compatible: bool,
                         error_type: Type[Exception]
                         ):

    def __init__(self):
        self.registry = {}
    def register_logging_callback(self,
                                   name: str,
                                   function: Callable[[str, Type[Exception]], None]
                                   ):
        if name not in self.allowed_keys:
            msg = f"""
            Attempted to setup logging callback that was not supported.
            
            Logging is expected to be one of {self.allowed_keys},
            But got: {name}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)

        self.registry[name] = function
    def __call__(self, config: str):
        if config not in self.allowed_keys:
            msg = f"""
            Attempted to get logging callback that was not
            defined.
            
            Logging is expected to be one of {self.allowed_keys},
            But got: {config}
            """
            msg = textwrap.dedent(msg)
            raise RuntimeError(msg)
        if config not in self.registry:
            msg = f"""
            Attempted to use validation mode that never had a 
            callback defined. It is likely you forgot to define the
            callback or file when using logging mode of 'callback' or
            'file'
            """
            msg = textwrap.dedent(msg)
            raise RuntimeError(msg)
        return self.registry[config]

class

logging_registry = LoggingRegistry()




# Specify some of the interations

class ValidationOperator:
    """
    When called upon, will execute a single validation
    check in an appropriate manner. Needs to be
    fleshed out
    """
    def __init__(self,
                 predicate_function: Callable[[Any], bool],
                 emission_function: Callable[[str], None]
                 ):
        self.predicate = predicate_function
        self.emission = emission_function
    def __call__(self, config:  operand: Any, ):

class Validator:
    """
    A class dedicated to making it easy to setup
    validation for a particular property.

    Implements as a prototype, that is then
    added onto. The user specifies the details of
    validation

    It is intended to be configured at the beginning of classes
    which will use it, and then called within the body.
    """
    general_message: str
    predicates: List[Callable[[Any], bool]]
    details: List[str]
    exception_types: List[Type[Exception]]
    def __init__(self, general_message: str):
        self.general_message = general_message
        self.predicates = []
        self.details = []
        self.exception_types = []

    def format_error_message(self,
                             details: str,
                             context: str)->str:
        """
        :param general_message: What validation was being performed
        :param details: What part of the validation failed
        :param context: Any context provided
        :return:
        """

        # Dedent so everything is on the same level
        general_message = textwrap.dedent(self.general_message)
        details = textwrap.dedent(details)
        context = textwrap.dedent(context)

        # Indent the details and the context
        details = textwrap.indent(details,"   ")
        context = textwrap.indent(context, "   ")

        # Put message together. Add context

        msg = general_message
        msg = msg + details
        msg = msg + "\n" + "Additional context avaliable is:"
        msg = msg + "\n" + context
        return msg

    def add_validation(self,
                      predicate: Callable[[Any], bool],
                      detailed_message: str,
                      error_type: Type[Exception],
                      ):
        """
        Adds a validation operation into the
        validator. This will consist of a predicate,
        and a more detailed error message.

        The error message will be raised when the
        predicate is true.

        :param predicate: A callable that accepts a validation operand and returns a bool
                          It should return true when it PASSES the test, like an assert
        :param detailed_message: A message with details on what is wrong that is raised
                            when the predicate fails
        :param error_type: The error to raise when in debug mode.
        """
        self.predicates.append(predicate)
        self.details.append(detailed_message)
        self.exception_types.append(error_type)

    def __call__(self, error_mode: str, context: str, operand: Any)->Any:

        for predicate, details, exception in zip(self.predicates, self.details, self.exception_types):
            if error_mode == ErrorModes.debug:
                if not predicate(operand):
                    msg = self.format_error_message(details, context)
                    raise exception(msg)
            elif error_mode == ErrorModes.checkify:
                msg = self.format_error_message(details, context)
                checkify.check(predicate(operand), msg)



