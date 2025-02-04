def upsample_sizes(a: int, b: int, factor: int) -> list[int]:
    range_values = []
    current_value = a
    while current_value <= b:
        range_values.append(current_value)
        current_value *= factor
    if range_values[-1] != b:
        range_values.append(b)
    return range_values