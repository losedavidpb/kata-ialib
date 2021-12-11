def get_decimal_format(value):
    if value == 0: return value
    f_value, num_decimals = format(value, '.4f'), 4

    while float(f_value) == 0:
        num_decimals = num_decimals + 1
        f_value = format(value, '.' + str(num_decimals) + 'f')

    return f_value

def format_file_path_extension(file_path, extension):
    if file_path is not None:
        if not file_path.endswith('.' + extension):
            file_path = file_path + '.' + extension

    return file_path
