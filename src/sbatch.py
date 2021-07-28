from jinja2 import Template
from shutil import which
from time import gmtime, strftime
import argparse
import fnmatch
import functools
import getpass
import logging
import operator
import os
import pwd
import re
import shlex
import signal
import socket
import subprocess
import sys
import time

"""
TODO:
  - More optional parameters
  - GPU utility
  - Merge error and output files
  - The log out/err should contain the actual script that was started
  - Calculate fairshare and priority
  - Optimize jupyter usage
  - Add the option for -exc and -inc for the this format tbm[0-5]
  - The NODE_SET is probably outdated
  - Put all constants in a config file
  - Clean up Jinja2 templates

Calculate fairshare = 0.50 * CPU + 0.25 * Mem[GB] + 2 * GPU) * walltime[sec]
Calculate priority = 40,000,000 * QoS + 20,000,000 * FairShare
"""

signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))


NODE_SET = set(
    [
        "tbm1",
        "tbm2",
        "tbm3",
        "tbm4",
        "insy1",
        "insy2",
        "insy3",
        "insy4",
        "insy5",
        "insy6",
        "insy7",
        "insy8",
        "insy11",
        "insy12",
        "insy13",
        "insy14",
        "insy15",
        "grs1",
        "grs2",
        "grs3",
        "grs4",
        "100plus",
        "ew1",
        "ewi2",
        "wis1",
    ]
)

TIME_DICT = {"short": "4:00:00", "long": "168:00:00", "infinite": "99999:00:00"}
CPU_DICT = {"short": 600, "long": 240, "infinite": 32}


DEFAULT_TASKS = 1
DEFAULT_CORES = 1
DEFAULT_PARTITION = "general"
DEFAULT_TIME = "1:00:00"
DEFAULT_QOS = "short"
DEFAULT_MEMORY = 1000
DEFAULT_JOB_NAME = "myjob"
DEFAULT_WORKDIR = "/../../user/"
DEFAULT_JUPYTER_LOG = "/../../jupyter/logs"

MAX_TASKS = 600
MAX_CPUS = 600
MAX_HOURS = 168
MAX_MEMORY = 750000

MIN_PORT = 8889

PREAMBLE = f"""# Created by job script generator for SLURM\n# {strftime("%a %b %d %H:%M:%S UTC %Y", gmtime())}\n"""

TO_EXECUTE_BEFORE = 'if [ "x$SLURM_JOB_ID" == "x" ]; then\n\techo "You need to submit your job to the queuing system with sbatch"\n\texit 1\nfi\n\nif [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]; then\n\techo "Submitted batch job $SLURM_JOB_ID on the node(s): $SLURM_NODELIST"\nfi\n\ndate -u'
TO_EXECUTE_AFTER = "date -u"


def run_job(sbatch_job, dur_sleep=0.05):
    ps = subprocess.Popen(("echo", sbatch_job), stdout=subprocess.PIPE)
    output = subprocess.check_output(
        ("sbatch"), stdin=ps.stdout, universal_newlines=True
    )
    print(output, end="")
    time.sleep(dur_sleep)
    ps.wait()


def kill_job(sbatch_job, message=None, dur_sleep=0.05):
    if message is None:
        message = f"Cancelling {sbatch_job}"

    ps = subprocess.Popen(("scancel", sbatch_job), stdout=subprocess.PIPE)
    # output = subprocess.check_output(
    #     ("sbatch"), stdin=ps.stdout, universal_newlines=True
    # )
    # out, err = ps.communicate()

    print(message)
    time.sleep(dur_sleep)
    ps.wait()


def count_prefix_queue(prefix):
    if which("squeue") is None:
        parser.error(
            f"The command `squeue` is not found on this system, cannot cancel jobs!"
        )

    user_name = pwd.getpwuid(os.getuid()).pw_name
    exec_str = (
        f'squeue -h -o "%.18i %.9P %.255j %.30u %.2t %.10M %R" -u {user_name}'
    )

    p = subprocess.Popen(
        shlex.split(exec_str), shell=False, bufsize=4096, stdout=subprocess.PIPE
    )
    out, err = p.communicate()
    jobs = [job.split() for job in out.decode("utf-8").strip().split("\n")]

    n = 0
    for job in jobs:
        if job:
            job_id, partition, job_name, job_user_name, job_status, job_time, job_node = (
                job
            )
            assert user_name == job_user_name
            if fnmatch.fnmatch(job_name, prefix):
                n += 1
    return n


def make_generic(cvalue, func_xs, comp_xs, typecast, message):
    """Function generator that can handle most of the different types needed by argparse

    Args:
        cvalue (T): Optional value to compare to with the argument of the inner function
        func_xs (List[func(x: T) -> bool]): Optional list value of functions that are applied onto the argument of the inner function
        comp_xs (List[func(x: T, y: T) -> bool] or List[func(x: T) -> bool]): List value of functions (two-argument if cvalue is set, otherwise only applied on the argument of the inner function)
        typecast (T): Typecast argument to cast the argument of the inner function
        message (str): fstring used in error messages should follow the format of {tvalue} {cvalue}
    Returns:
        f(x: T) -> x: T
    """

    def check_generic(value):
        if func_xs:
            tvalue = functools.reduce(lambda res, f: f(res), func_xs, typecast(value))
        else:
            tvalue = typecast(value)

        if cvalue is None:
            if functools.reduce(lambda res, f: f(res), comp_xs, tvalue):
                raise argparse.ArgumentTypeError(message.format(tvalue, cvalue))
        else:
            if any([func(tvalue, cvalue) for func in comp_xs]):
                raise argparse.ArgumentTypeError(message.format(tvalue, cvalue))

        return typecast(value)

    return check_generic


def check_time(value):
    if not re.match(
        "^(([1-9]|0[0-9]|1[0-9]|[0-9][0-9]|1[0-6][0-8]):([0-5][0-9]):([0-5][0-9]))$",
        value,
    ):
        raise argparse.ArgumentTypeError(
            f"Invalid time provided: {value}, example format: 12:30:15, should be between 0 to {MAX_HOURS} hours"
        )
    hours, minutes, seconds = map(int, value.split(":"))
    if hours == MAX_HOURS and (minutes != 0 or seconds != 0):
        raise argparse.ArgumentTypeError(
            f"{value} invalid time provided, should be between 0 to {MAX_HOURS} hours"
        )
    return value


def check_output(value):
    if value:
        if not re.search(".out$", value):
            raise argparse.ArgumentTypeError(
                f"Invalid output file specified, should end with .out! {value}"
            )
        value = "".join(["%j-", ".".join(value.split(".")[:-1]), ".out"])
    return value


def check_error(value):
    if value:
        if not re.search(".err$", value):
            raise argparse.ArgumentTypeError(
                f"Invalid error file specified, should end with .err! {value}"
            )
        value = "".join(["%j-", ".".join(value.split(".")[:-1]), ".err"])
    return value


def check_workdir(value):
    # return os.path.join(value, '')  # Stuck with os.path since pathlib likes to strip trailing slashes
    if os.path.isdir(value):
        return os.path.join(value, "")
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid/nonexistant working directory given! {value}"
        )


def calc_fairshare(cpu, mem, time, gpu=0):
    # TODO: Update fairshare function
    return int((0.50 * cpu + 0.125 * (mem / 1000.0) + 10 * gpu) * time_to_seconds(time))


def time_to_seconds(td):
    return sum(x * int(t) for x, t in zip([3600, 60, 1], td.split(":")))


def compare_time(t1, t2):
    return time_to_seconds(t1) > time_to_seconds(t2)


def sanity_checks(parser, args):
    if compare_time(args.time, TIME_DICT[args.qos]):
        parser.error(
            f"Maximum time for specified qos='{args.qos}' is limited to {TIME_DICT[args.qos]}, attempted to allocate {args.time}"
        )

    if args.cpus > CPU_DICT[args.qos]:
        parser.error(
            f"CPU usage of qos '{args.qos}' is limited to {CPU_DICT[args.qos]}, attempted to allocate {args.cpus}"
        )

    # TODO: Clean up
    if args.exclude:
        args.exclude = ",".join(args.exclude)

    if args.include:
        args.include = ",".join(args.include)

    # Workaround to get newlines working TODO: Figure something more elegant
    if vars(args).get("script"):
        args.script = "\n".join(args.script.split("\\n"))

    fairshare = calc_fairshare(args.cpus, args.mem, args.time)
    logger.warning(f"Fairshare usage: {fairshare}")


def make_job(
    partition=DEFAULT_PARTITION,
    qos=DEFAULT_QOS,
    time=DEFAULT_TIME,
    ntasks=DEFAULT_TASKS,
    cpus=DEFAULT_CORES,
    mem=DEFAULT_MEMORY,
    jobname=DEFAULT_JOB_NAME,
    workdir=DEFAULT_WORKDIR,
    output="",
    error="",
    incerror=None,
    script="",
    constraints="",
    include="",
    exclude="",
):

    execute_before = TO_EXECUTE_BEFORE
    execute_after = TO_EXECUTE_AFTER

    template = Template(
        "#!/bin/sh\n\
{{ preamble }}\n\
#SBATCH --partition={{ partition }}\n\
#SBATCH --qos={{ qos }}\n\
#SBATCH --time={{ time }}\n\
#SBATCH --ntasks={{ ntasks }}\n\
#SBATCH --cpus-per-task={{ cpus }}\n\
#SBATCH --mem={{ mem }}\n\
#SBATCH --job-name={{ jobname }}\n\
#SBATCH --chdir={{ workdir }}\n\
{% if constraints %}#SBATCH --constraint={{constraints}}\n{% endif %}\
{% if include %}#SBATCH --nodelist={{include}}\n{% endif %}\
{% if exclude %}#SBATCH --exclude={{exclude}}\n{% endif %}\
#SBATCH --output={% if output %}{{output}}{% else %}%j-{{jobname}}.out {% endif %}\n\
{% if error %}#SBATCH --error={{error}}\n{% elif incerror %}#SBATCH --error=%j-{{jobname}}.err\n {% endif %}\
\n\
{{ execute_before }}\
\n{{ script }}\n\
{{ execute_after }}"
    )

    return template.render(locals())


def make_jupyter_job(
    partition=DEFAULT_PARTITION,
    qos=DEFAULT_QOS,
    time=DEFAULT_TIME,
    ntasks=DEFAULT_TASKS,
    cpus=DEFAULT_CORES,
    mem=DEFAULT_MEMORY,
    jobname=DEFAULT_JOB_NAME,
    workdir=DEFAULT_WORKDIR,
    port=8889,
    output="",
    error="",
    incerror=None,
    constraints="",
    include="",
    exclude="",
):

    execute_before = TO_EXECUTE_BEFORE
    execute_after = TO_EXECUTE_AFTER
    jupyter_log = DEFAULT_JUPYTER_LOG
    user = getpass.getuser()

    template = Template(
        """#!/bin/sh\n\
{{ preamble }}\n\
#SBATCH --partition={{ partition }}\n\
#SBATCH --qos={{ qos }}\n\
#SBATCH --time={{ time }}\n\
#SBATCH --ntasks={{ ntasks }}\n\
#SBATCH --cpus-per-task={{ cpus }}\n\
#SBATCH --mem={{ mem }}\n\
#SBATCH --job-name={{ jobname }}\n\
#SBATCH --chdir={{ workdir }}\n\
{% if constraints %}#SBATCH --constraint={{constraints}}\n{% endif %}\
{% if include %}#SBATCH --nodelist={{include}}\n{% endif %}\
{% if exclude %}#SBATCH --exclude={{exclude}}\n{% endif %}\
#SBATCH --output={% if output %}{{output}}{% else %}{{jupyter_log}}%j-{{jobname}}.out {% endif %}\n\
{% if error %}#SBATCH --error={{error}}\n{% elif incerror %}#SBATCH --error=%j-{{jobname}}.err\n {% endif %}\
\n\
if [ "x$SLURM_JOB_ID" == "x" ]; then\n\
\techo "You need to submit your job to the queuing system with sbatch"\n\
\texit 1\nfi\n\n\
if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]; then\n\
\techo "Submitted batch job $SLURM_JOB_ID on the node(s): $SLURM_NODELIST"\n\
fi\n\n\
XDG_RUNTIME_DIR=""\n\
node=$(hostname)\n\
echo -e "\
Command to create ssh tunnel:\n\
ssh -N -f -L {{port}}:${node}:{{port}} {{user}}@login1.hpc.tudelft.nl -o ProxyCommand='ssh -W %h:%p {{user}}@linux-bastion.tudelft.nl'\n\
Use a Browser on your local machine to go to:\n\
http://localhost:{{port}} (prefix w/ https:// if using password)"\n\
date -u\n\
jupyter-lab --no-browser --port={{port}} --ip=0.0.0.0\n\
{{ TO_EXECUTE_AFTER }}"""
    )

    return template.render(locals())


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    def cmd_gen(args):
        sanity_checks(parser, args)

        job_script = make_job(
            partition=args.partition,
            qos=args.qos,
            time=args.time,
            ntasks=args.ntasks,
            cpus=args.cpus,
            mem=args.mem,
            jobname=args.jobname,
            workdir=args.workdir,
            output=args.output,
            error=args.error,
            incerror=args.incerror,
            script=args.script,
            constraints=args.constraints,
            include=args.include,
            exclude=args.exclude,
        )
        if args.sbatch:
            run_job(job_script)
        else:
            print(job_script)

    def cmd_jupyter(args):
        sanity_checks(parser, args)

        job_script = make_jupyter_job(
            partition=args.partition,
            qos=args.qos,
            time=args.time,
            ntasks=args.ntasks,
            cpus=args.cpus,
            mem=args.mem,
            jobname=args.jobname,
            workdir=args.workdir,
            port=args.port,
            incerror=args.incerror,
            constraints=args.constraints,
            include=args.include,
            exclude=args.exclude,
        )

        if args.sbatch:
            run_job(job_script)
        else:
            print(job_script)

    def cmd_cancel(args):
        if which("squeue") is None:
            parser.error(
                f"The command `squeue` is not found on this system, cannot cancel jobs!"
            )

        user_name = pwd.getpwuid(os.getuid()).pw_name
        exec_str = (
            f'squeue -h -o "%.18i %.9P %.255j %.30u %.2t %.10M %R" -u {user_name}'
        )

        p = subprocess.Popen(
            shlex.split(exec_str), shell=False, bufsize=4096, stdout=subprocess.PIPE
        )
        out, err = p.communicate()
        jobs = [job.split() for job in out.decode("utf-8").strip().split("\n")]

        for job in jobs:
            job_id, partition, job_name, job_user_name, job_status, job_time, job_node = (
                job
            )
            assert user_name == job_user_name
            if args.name:
                if fnmatch.fnmatch(job_name, args.name):
                    kill_job(
                        job_id, message=f"Cancelling {job_name} with job id {job_id}"
                    )

    check_memory = make_generic(
        MAX_MEMORY,
        None,
        [operator.ge, lambda *x: x[0] <= 0],
        int,
        '"{}" invalid amount of memory provided, should be between 1 to {} MB',
    )
    check_cpus = make_generic(
        MAX_CPUS,
        None,
        [operator.gt, lambda *x: x[0] <= 0],
        int,
        '"{}" is not a valid number of CPUs (range is [1, {}] CPUs)',
    )
    check_tasks = make_generic(
        MAX_TASKS,
        None,
        [operator.gt, lambda *x: x[0] <= 0],
        int,
        '"{}" is not a valid number of tasks (range is  [1, {}] tasks)',
    )
    check_port = make_generic(
        MIN_PORT, None, [operator.lt], int, '"{}" is not a valid port (must be >={})'
    )

    check_clude = make_generic(
        None,
        None,
        [lambda x: x not in NODE_SET],
        str,
        f"One of the selected nodes is not in the node list: [{', '.join(NODE_SET)}]",  # TODO: Cleanup
    )

    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        epilog="""
examples:
  %(prog)s generate -s 'echo \"test\"' -m 4000 -t 4:00:00 -j test
  %(prog)s jupyter -pt 9001
  %(prog)s cancel -n \"stop_me_4*\" # Kill all jobs starting with the 'stop_me_4' prefix""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    parser_logger = argparse.ArgumentParser(add_help=False)

    parser_logger.add_argument(
        "-l",
        dest="log_level",
        help="Set the logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    parser_parent = argparse.ArgumentParser(add_help=False)

    parser_parent.add_argument(
        "-p",
        "--partition",
        dest="partition",
        help="Partition to run on",
        choices=["general"],
        default=DEFAULT_PARTITION,
        type=str,
    )

    parser_parent.add_argument(
        "-q",
        "--qos",
        dest="qos",
        help="qos to run on",
        choices=["short", "long", "infinite"],
        default=DEFAULT_QOS,
        type=str,
    )

    parser_parent.add_argument(
        "-m",
        "--mem",
        dest="mem",
        help="Allocate amount of memory (in MB)",
        default=DEFAULT_MEMORY,
        type=check_memory,
    )
    parser_parent.add_argument(
        "-c",
        "--cpus-per-task",
        dest="cpus",
        help="Allocate number of cpus",
        default=DEFAULT_CORES,
        type=check_cpus,
    )
    parser_parent.add_argument(
        "-n",
        "--ntasks",
        dest="ntasks",
        help="Allocate number of tasks",
        default=DEFAULT_TASKS,
        type=check_tasks,
    )

    parser_parent.add_argument(
        "-j",
        "--job-name",
        dest="jobname",
        help="Specify name of the job",
        default=DEFAULT_JOB_NAME,
        type=str,
    )

    parser_parent.add_argument(
        "-w",
        "--workdir",
        dest="workdir",
        help="Specify working directory",
        default=DEFAULT_WORKDIR,
        type=check_workdir,
    )

    parser_parent.add_argument(
        "-t",
        "--time",
        dest="time",
        help="Time to allocate",
        default=DEFAULT_TIME,
        type=check_time,
    )

    parser_parent.add_argument(
        "-ie",
        "--incerror",
        dest="incerror",
        help="Flag whether error is included or not",
        action="store_true",
    )

    parser_parent.add_argument(
        "-d",
        "--constraints",
        dest="constraints",
        help="Set processor architecture constraints",
        type=str,
        choices=set(["avx", "avx2"]),
        default=None,
    )

    parser_parent.add_argument(
        "--sbatch",
        dest="sbatch",
        help="Run after generating the script",
        action="store_true",
    )

    parser_parent.add_argument(
        "-exc", "--exclude", dest="exclude", type=check_clude, nargs="+"
    )

    parser_parent.add_argument(
        "-inc", "--include", dest="include", type=check_clude, nargs="+"
    )

    ########### Generate parser

    parser_gen = subparsers.add_parser(
        "generate",
        help="Create and run sbatch scripts",
        parents=[parser_parent, parser_logger],
    )
    parser_gen.set_defaults(func=cmd_gen)

    parser_gen.add_argument(
        "-s",
        "--script",
        dest="script",
        help="Job string to run (in single quotes)",
        type=str,
        required=True,
    )

    parser_gen.add_argument(
        "-o", "--output", dest="output", default=None, type=check_output
    ),

    parser_gen.add_argument(
        "-e", "--error", dest="error", default=None, type=check_error
    )

    ########### Jupyter parser

    parser_jupyter = subparsers.add_parser(
        "jupyter",
        help="Create and run a jupyter notebook/lab",
        parents=[parser_parent, parser_logger],
    )
    parser_jupyter.set_defaults(func=cmd_jupyter)

    parser_jupyter.add_argument(
        "-pt",
        "--port",
        dest="port",
        help="Port to use",
        default="8889",
        type=check_port,
    )

    ########### Cancel parser

    parser_cancel = subparsers.add_parser(
        "cancel",
        help="Cancel sbatch jobs that are in the queue",
        parents=[parser_logger],
    )
    parser_cancel.set_defaults(func=cmd_cancel)

    parser_cancel.add_argument(
        "-n",
        dest="name",
        help="Cancel based on name (wildcard supported, make sure to provide in double quotes)",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    if args.log_level:
        print(args.log_level, logging)
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    args.func(args)
