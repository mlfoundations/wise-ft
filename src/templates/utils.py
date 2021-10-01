
def get_plural(name):
    name = name.replace('_', ' ')
    if name[-2:] == 'sh':
        name = name + 'es'
    elif name[-2:] == 'ch':
        name = name + 'es'
    elif name[-1:] == 'y':
        name = name[:-1] + 'ies'
    elif name[-1:] == 's':
        name = name + 'es'
    elif name[-1:] == 'x':
        name = name + 'es'
    elif name[-3:] == 'man':
        name = name[:-3] + 'men'
    elif name == 'mouse':
        name = 'mice'
    elif name[-1:] == 'f':
        name = name[:-1] + 'ves'
    else:
        name = name + 's'
    return name


def append_proper_article(name):
    name = name.replace('_', ' ')
    if name[0] in 'aeiou':
        return 'an ' + name
    return 'a ' + name
