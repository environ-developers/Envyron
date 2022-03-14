from argparse import ArgumentParser

from .input import Input

parser = ArgumentParser()

parser.add_argument(
    '-n',
    metavar='natoms',
    dest='natoms',
    help='Number of atoms',
    type=int,
    default=1,
)

parser.add_argument(
    '-f',
    metavar='filename',
    dest='filename',
    help='Input file name',
    type=str,
    default=None,
)

parser.add_argument(
    '--debug',
    action='store_true',
    help='Turn on debugging output',
)

args = vars(parser.parse_args())

if not args['debug']:
    import sys
    sys.tracebacklimit = 0

my_input = Input(args['natoms'], args['filename'])

for section in my_input:
    name, fields = section

    if fields:

        print(f"\n{name}\n")

        for field in fields:
            name, value = field

            if name == 'functions':

                for i, group in enumerate(value, 1):
                    print(f"\ngroup {i}")

                    for function in group:
                        print(function)

            else:
                print(f"{name} = {value}")

print()
