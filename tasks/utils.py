import os

work_dir = os.path.join(os.path.dirname(__file__), "../runs")


def create_no_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_workspace(task_name):
    create_no_exists(work_dir)
    workspace = os.path.join(work_dir, task_name)
    create_no_exists(workspace)
    log_dir = os.path.join(workspace, "logs")
    create_no_exists(log_dir)
    models_dir = os.path.join(workspace, "model_files")
    create_no_exists(models_dir)
    return task_name, workspace, log_dir, models_dir
