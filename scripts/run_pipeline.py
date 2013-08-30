

import argparse
import os
import os.path
import shutil

import yaml

from drift.core import manager
from drift.pipeline import pipeline, timestream


def run_config(args):

    pl = pipeline.PipelineManager.from_configfile(args.configfile)
    pl.simulate()
    pl.generate()

pipe = None
def interactive_config(args):
    global pipe
    pipe = pipeline.PipelineManager.from_configfile(args.configfile)




def queue_config(args):

    yconf = yaml.safe_load(args.configfile)

    ## Global configuration
    ## Create output directory and copy over params file.
    if 'config' not in yconf:
        raise Exception('Configuration file must have an \'config\' section.')

    conf = yconf['config'] 

    outdir = conf['output_directory'] if 'output_directory' in conf else conf['timestream_directory']
    outdir = os.path.normpath(os.path.expandvars(os.path.expanduser(outdir)))

    if not os.path.isabs(outdir):
        raise Exception("Output directory path must be absolute.")

    pbsdir = os.path.normpath(outdir + '/pbs/')

    # Create directory if required
    if not os.path.exists(pbsdir):
        os.makedirs(pbsdir)

    # Copy config file into output directory (check it's not already there first)
    sfile = os.path.realpath(os.path.abspath(args.configfile.name))
    dfile = os.path.realpath(os.path.abspath(pbsdir + '/config.yaml'))

    if sfile != dfile:
        shutil.copy(sfile, dfile)


    conf['mpiproc'] = conf['nodes'] * conf['pernode']
    conf['pbsdir'] = pbsdir
    conf['scriptpath'] = os.path.realpath(__file__)


    script="""#!/bin/bash
#PBS -l nodes=%(nodes)i:ppn=%(ppn)i
#PBS -q %(queue)s
#PBS -r n
#PBS -m abe
#PBS -V
#PBS -l walltime=%(time)s
#PBS -N %(name)s


cd %(pbsdir)s
export OMP_NUM_THREADS=%(ompnum)i

mpirun --mca btl self,sm,tcp -np %(mpiproc)i -npernode %(pernode)i python %(scriptpath)s run config.yaml &> jobout.log
"""

    script = script % conf

    scriptname = pbsdir + "/jobscript.sh"

    with open(scriptname, 'w') as f:
        f.write(script)

    if not args.nosubmit:
        os.system('cd %s; qsub jobscript.sh' % pbsdir)




parser = argparse.ArgumentParser(description='Run/queue the pipeline to analyse a timestream.')
subparsers = parser.add_subparsers(help='Command to run.', title="Commands", metavar="<command>")


parser_run = subparsers.add_parser('run', help='Run the analysis from the given config file.')
parser_run.add_argument('configfile', type=str, help='Configuration file to use.')
parser_run.set_defaults(func=run_config)


parser_interactive = subparsers.add_parser('interactive', help='Interactive analysis with the given config file.')
parser_interactive.add_argument('configfile', type=str, help='Configuration file to use.')
parser_interactive.set_defaults(func=interactive_config)


parser_queue = subparsers.add_parser('queue', help='Create a jobscript for running the pipeline and add to the PBS queue.')
parser_queue.add_argument('configfile', type=argparse.FileType('r'), help='Configuration file to use.')
parser_queue.add_argument('--nosubmit', action='store_true', help='Don\'t submit the job to the queue.')
parser_queue.set_defaults(func=queue_config)

args = parser.parse_args()
args.func(args)





