def aae(actual: float, expected: float, places: int = 3):
    """Assert that two floats are equal up to a specified number of decimal places."""
    assert actual is not None, "Actual result is None"
    assert expected is not None, "Expected result is None"
    rounded_actual = round(actual, places)
    rounded_expected = round(expected, places)
    assert rounded_actual == rounded_expected, (
        f"Assertion failed: {rounded_actual} != {rounded_expected} "
        f"(rounded to {places} places)"
    )
