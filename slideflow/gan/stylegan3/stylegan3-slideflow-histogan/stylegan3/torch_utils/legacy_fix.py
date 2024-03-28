def apply_legacy_patch(src):

    def wrap_import(line):
        if line.startswith('from ..torch_utils'):
            return ('try:'
                    f'\n\t{line}'
                    '\nexcept:'
                    f'\n\t{line.replace("..torch_utils", "torch_utils")}')
        else:
            return line

    return '\n'.join([wrap_import(line) for line in src.split('\n')])
