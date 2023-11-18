import argparse
import os
import glob, shutil
from pathlib import Path

MODELS = {
    "spanbert": f"/gpfswork/rech/pds/upa43yu/models/spanbert-base-cased",
    "bert": f"/gpfswork/rech/pds/upa43yu/models/bert-base-cased",
    "roberta": f"/gpfswork/rech/pds/upa43yu/models/roberta-base",
    "scibert": f"/gpfswork/rech/pds/upa43yu/models/scibert-base",
    "arabert": f"/gpfswork/rech/pds/upa43yu/models/bert-base-arabert",
    "bertlarge": f"/gpfsdswork/dataset/HuggingFace_Models/bert-large-cased",
    "scibert_cased": f"/gpfswork/rech/pds/upa43yu/models/scibert_cased",
    "albert": f"/gpfswork/rech/pds/upa43yu/models/albert-xxlarge-v2",
    "spanbertlarge": f"/gpfswork/rech/pds/upa43yu/models/spanbert-large-cased",
    "t5-s": "/gpfsdswork/dataset/HuggingFace_Models/t5-small",
    "t5-m": "/gpfsdswork/dataset/HuggingFace_Models/t5-base",
    "t5-l": "/gpfsdswork/dataset/HuggingFace_Models/t5-large",
    "deberta": "/gpfswork/rech/pds/upa43yu/models/deberta-v3-large"
}


def create_parser():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the config file')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args_job = parser.parse_args()

    # Load config file
    import yaml

    with open(args_job.config, 'r') as f:
        config = yaml.safe_load(f)

        # convert config to namespace
        from argparse import Namespace

        args = Namespace(**config)

    gpu = "#SBATCH -A " + args.gpu  # "#SBATCH -A pds@v100" "#SBATCH -A pds@a100"

    if "v100" in gpu:
        constraint = "#SBATCH -C v100-32g"
    elif "a100" in gpu:
        constraint = "#SBATCH -C a100"
    # constraint = args.constraint  # "#SBATCH -C v100-32g" "#SBATCH -C a100"
    run_time = args.run_time

    root_dir = os.path.join(os.getcwd(), args.root_dir)

    # create root log directory
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    args.name = args.model_name.split("/")[-1]

    # args.log_dir seed is increment by 1 of the previous one
    start = 0
    while True:
        logs_dir = f"{args.name}_{start}"
        if not os.path.exists(os.path.join(root_dir, logs_dir)):
            break
        start += 1

    args.log_dir = logs_dir

    # create log directory
    log_dir = root_path / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / '.out').mkdir(exist_ok=True)

    # convert to string
    args.log_dir = str(log_dir)

    job_directory = log_dir / '.job'
    job_directory.mkdir(exist_ok=True)

    # python files in the current directory
    # create a directory in the log directory and copy all the python files
    name = "python_files"
    store = log_dir / name
    store.mkdir(exist_ok=True)
    for file in glob.glob("*.py"):
        shutil.copy(file, store)

    log_name = f"{args.name}_{start}"

    # copy the config file in the log directory
    shutil.copy(args_job.config, log_dir)

    # copy the directory "modules" of the model in the log directory (log_dir/python_files)
    shutil.copytree("modules", os.path.join(log_dir, "python_files/modules"))

    log_content = f"""#!/bin/bash\n
#SBATCH --job-name={log_name}\n
#SBATCH --output={log_dir}/.out/{log_name}.out\n
#SBATCH --error={log_dir}/.out/{log_name}.err\n
{gpu}\n
{constraint}\n
#SBATCH --time={run_time}\n
#SBATCH --nodes=1\n
#SBATCH --ntasks=1\n
#SBATCH --gres=gpu:1\n
#SBATCH --cpus-per-task=10\n
#SBATCH --hint=nomultithread\n
source "/gpfswork/rech/pds/upa43yu/miniconda/bin/activate"\n
python train.py --config {args_job.config} --log_dir {log_dir}"""

    job_file = os.path.join(job_directory, f"{log_name}.job")

    # print("Job file: ", job_file)
    # print("job_directory: ", job_directory)

    with open(job_file, "w") as fh:
        print("Writing job file: ", job_file)
        fh.write(log_content)

    os.system(f"sbatch {job_file}")
