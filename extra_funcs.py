# Gelin Eguinosa Rosique
# 2022

from sys import stdout
from time import time, ctime


def progress_bar(progress, total):
    """
    Print on the console a bar representing the progress of the function that
    called this method.

    Args:
        progress: How many of the planned actions the program has done.
        total: The total amount of actions the program needs to do.
    """
    # Values to print.
    steps = progress * 40 // total
    percentage = progress * 100 // total

    # Print progress bar.
    stdout.write('\r')
    stdout.write("[%-40s] %03s%%" % ('=' * steps, percentage))
    stdout.flush()

    # Add Break at the end of the progress bar, if we are done.
    if progress == total:
        print()


def progress_msg(message):
    """
    Display the provided 'message' to the user about the progress of the
    program.

    Args:
        message: String with the message we have to display.
    """
    print(f"[{time_info()}] {message}")


def time_info():
    """
    Create a string with the time data in the format hh:mm:ss.mss.

    Returns: String with the time data.
    """
    current_time = time()
    millisecond = int((current_time - int(current_time)) * 1000)
    # Format HH:MM:SS.MSS
    time_data = ctime(current_time)[11:19] + '.' + number_to_3digits(millisecond)
    return time_data


def big_number(number):
    """
    Add commas to number with more than 3 digits, so they are more easily read.

    Args:
        number: The number we want to transform to string

    Returns:
        The string of the number with the format: dd,ddd,ddd,ddd
    """
    # Get the string of the number.
    number_string = str(number)

    # Return its string if it's not big enough.
    if len(number_string) <= 3:
        return number_string

    # Add the commas.
    new_string = number_string[-3:]
    number_string = number_string[:-3]
    while len(number_string) > 0:
        new_string = number_string[-3:] + ',' + new_string
        number_string = number_string[:-3]

    # Return the reformatted string of the number.
    return new_string


def number_to_3digits(number):
    """
    Transform a number smaller than 1000 (0-999) to a string representation with
    three characters (000, 001, ..., 021, ..., 089, ..., 123, ..., 999).
    """
    # Make sure the value we transform is under 1000 and is positive.
    mod_number = number % 1000
    
    if mod_number < 10:
        return "00" + str(mod_number)
    elif mod_number < 100:
        return "0" + str(mod_number)
    else:
        return str(mod_number)


def number_to_digits(number, digits: int):
    """
    Transform a number to a number to a string representation with at least
    'digits' number of characters before the decimal point. For example, if
    number=1.23 and digits=3 then << result = '001.23' >>.
    """
    # Check we have a valid value for 'digits'.
    if digits <= 1:
        return str(number)
    # Iterate through all the powers of 10 (10, 100, 1000, ...)
    ten_powers = [10 ** x for x in range(1, digits)]
    ten_powers.reverse()
    prefix = ''
    for ten_power in ten_powers:
        # See if we need to add a zero to the prefix.
        if number < ten_power:
            prefix += '0'
    # String of the number with the required digits in the whole part.
    number_str = prefix + str(number)
    return number_str


def number_to_size(number: int, size: int):
    """
    Transform 'number' to a string of the given 'size', using zeros to the left
    to complete the desired size. The format of the numbers will be in the
    following form:
     - 01_000 (number=1_000, size=5)
     - 033 (number=33, size=3)
    """
    # Format with underscores the normal number.
    n_size = len(str(number))
    number_str = big_number(number).replace(',', '_')

    # Check if we have to add zeros to the left.
    if n_size > size:
        return number_str
    # Add zeros to the left.
    for _ in range(size - n_size):
        number_str = '0' + number_str
    return number_str


if __name__ == '__main__':
    # # My own test.
    # _number = 0.2
    # _digits = 1
    # # Use method.
    # _final_str = number_to_digits(_number, _digits)
    # print(f"Final Number: {_final_str}")

    # Testing Numbers to Digits.
    print("\n(To quit enter q/quit)")
    while True:
        # Get Number.
        _input = input("\nNumber: ")
        if _input in {'', 'q', 'quit'}:
            break
        _number = float(_input)
        # Get Digits.
        _input = input("Digits: ")
        if _input in {'', 'q', 'quit'}:
            break
        _digits = int(_input)
        # Use method.
        _final_str = number_to_digits(_number, _digits)
        print(f"Final Number: {_final_str}")
