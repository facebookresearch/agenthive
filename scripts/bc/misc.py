
def parse_overrides(config, overrides):
    """
    Overrides the values specified in the config with values.
    config: (Nested) dictionary of parameters
    overrides: Parameters to override and new values to assign. Nested
        parameters are specified via dot notation.
    >>> parse_overrides({}, [])
    {}
    >>> parse_overrides({}, ['a'])
    Traceback (most recent call last):
      ...
    ValueError: invalid override list
    >>> parse_overrides({'a': 1}, [])
    {'a': 1}
    >>> parse_overrides({'a': 1}, ['a', 2])
    {'a': 2}
    >>> parse_overrides({'a': 1}, ['b', 2])
    Traceback (most recent call last):
      ...
    KeyError: 'b'
    >>> parse_overrides({'a': 0.5}, ['a', 'test'])
    Traceback (most recent call last):
      ...
    ValueError: could not convert string to float: 'test'
    >>> parse_overrides(
    ...    {'a': {'b': 1, 'c': 1.2}, 'd': 3},
    ...    ['d', 1, 'a.b', 3, 'a.c', 5])
    {'a': {'b': 3, 'c': 5.0}, 'd': 1}
    """
    if len(overrides) % 2 != 0:
        # print('Overrides must be of the form [PARAM VALUE]*:', ' '.join(overrides))
        raise ValueError('invalid override list')

    for param, value in zip(overrides[::2], overrides[1::2]):
        keys = param.split('.')
        params = config
        for k in keys[:-1]:
            if k not in params:
                raise KeyError(param)
            params = params[k]
        if keys[-1] not in params:
            raise KeyError(param)

        current_type = type(params[keys[-1]])
        value = current_type(value)  # cast to existing type
        params[keys[-1]] = value

    return config