import argparse
import os
import sys


class TeeStderr(object):
    def __init__(self, name, mode='a'):
        self.file = open(name, mode)
        self.stderr = sys.stderr
        sys.stderr = self

    def __del__(self):
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stderr.write(data)

    def flush(self):
        self.file.flush()

class TeeStdout(object):
    def __init__(self, name, mode='a'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

class TeeAll(object):
    def __init__(self, name, mode='a'):
        os.makedirs(os.path.dirname(name), exist_ok=True)
        self.tee_stdout = TeeStdout(name, mode)
        self.tee_stderr = TeeStderr(name, mode)

class ArgumentSaver:
    def __init__(self):
        self.arguments = {}

    def add_argument(self, *args, **kwargs):
        arg_0 = args[0]
        arg_name = arg_0[2:] if arg_0.startswith("--") else arg_0[1:] if arg_0.startswith("-") else arg_0
        self.arguments[arg_name] = (args, kwargs)

    def add_to_parser(self, parser,exclude=None):
        for arg_name, (args, kwargs) in self.arguments.items():
            if exclude is not None and arg_name in exclude:
                continue
            parser.add_argument(*args, **kwargs)

class AddOutFileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

        outfile = os.path.join(values, os.path.basename(values)+'.out')
        setattr(namespace, 'outfile', outfile)
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')