import textwrap
from enum import Enum
from typing import Callable, Tuple, List, Any, Type
from jax.experimental import checkify

class ErrorModes(Enum):
    """
    The three error modes classes promise to
    operate under. They may be silent,
    checkify, or debug

    - silent: I will not raise errors at all.
    - checkify: I will raise errors through the checkify library.
    - debug: I will raise errors immediately. This might not be jit
             compatible, though.
    """

    silent = 'silent'
    fix = 'fix'
    checkify = 'checkify'
    debug = 'debug'

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



