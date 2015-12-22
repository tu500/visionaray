import re
import unicodedata


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.

    Adapted from the django project. (https://www.djangoproject.com/)
    """
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    return re.sub('[-\s]+', '-', value)

class Variant():

    def __init__(self, name, cmake_flags=None, *args, make_targets=None, fpstest_binary=None, foldername=None):

        self.name = name
        self.cmake_flags = cmake_flags
        self.make_targets = make_targets
        self.fpstest_binary = fpstest_binary
        self.foldername = foldername

        if self.cmake_flags is None:
            self.cmake_flags = []
        if type(self.cmake_flags) == str:
            self.cmake_flags = [self.cmake_flags]

        if self.make_targets is None:
            self.make_targets = []
        if type(self.make_targets) == str:
            self.make_targets = [self.make_targets]

        if foldername is None:
            self.foldername = slugify(name)

def restriction_applies(variant_line, restriction):

    if restriction is None:
        # No restriction
        return True

    # turn 'a,b+c' into
    # [ ['a'],
    #   ['b','c'] ]
    restriction_list = restriction.split(',')
    restriction_list = [i.split('+') for i in restriction_list]

    for restriction_item in restriction_list:

        contains_all_tokens = True
        for token in restriction_item:
            if not token in variant_line:
                contains_all_tokens = False
                break

        if contains_all_tokens == True:
            return True

    # No restriction matched
    return False


def specialize_variant_list(vlist, new_variants):

    if not vlist:
        vlist = [()]

    out = []

    for variant_line in vlist:

        specialized = False

        for v in new_variants:

            if type(v) == str:
                # variant given is only a string -> no restrictions
                name = v
                restrict_to = None
            else:
                name, restrict_to = v

            if restriction_applies(variant_line, restrict_to):
                out.append(variant_line + tuple(name.split('+')))
                specialized = True

        if not specialized:
            out.append(variant_line)

    return out

# TODO this one has the problem of build directory collision
# i.e. projects will be built in 'original_dir' and 'original_dir/extension'
def extend_variant_list(vlist, new_variants):

    out = []
    out.extend(vlist)
    out.extend(specialize_variant_list(vlist, new_variants))
    return out
