
def format_float_string(num):
    """Formats a float to a string with significant digits."""
    s = "{:.6f}".format(num)
    return s.rstrip('0').rstrip('.') if '.' in s else s
