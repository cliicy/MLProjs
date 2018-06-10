import mlflow
import os
import yaml
import tempfile
import hashlib
import sys
import json
import shutil
from six.moves import shlex_quote
from mlflow import data
import mlflow.tracking as tracking
from mlflow.entities.source_type import SourceType
from mlflow.entities.param import Param
from mlflow.utils import process, rest_utils

class ExecutionException(Exception):
    pass

class Project(object):
    """A project specification loaded from an MLproject file."""
    def __init__(self, uri, yaml_obj):
        self.uri = uri
        self.name = os.path.splitext(os.path.basename(os.path.abspath(uri)))[0]
        self.conda_env = yaml_obj.get("conda_env")
        self.entry_points = {}
        for name, entry_point_yaml in yaml_obj.get("entry_points", {}).items():
            parameters = entry_point_yaml.get("parameters", {})
            command = entry_point_yaml.get("command")
            self.entry_points[name] = EntryPoint(name, parameters, command)
        # TODO: validate the spec, e.g. make sure paths in it are fine

    def get_entry_point(self, entry_point):
        if entry_point in self.entry_points:
            return self.entry_points[entry_point]
        _, file_extension = os.path.splitext(entry_point)
        ext_to_cmd = {".py": "python", ".sh": os.environ.get("SHELL", "bash")}
        if file_extension in ext_to_cmd:
            command = "%s %s" % (ext_to_cmd[file_extension], shlex_quote(entry_point))
            return EntryPoint(name=entry_point, parameters={}, command=command)
        raise ExecutionException("Could not find {0} among entry points {1} or interpret {0} as a "
                                 "runnable script. Supported script file extensions: "
                                 "{2}".format(entry_point, list(self.entry_points.keys()),
                                              list(ext_to_cmd.keys())))


class EntryPoint(object):
    """An entry point in an MLproject specification."""
    def __init__(self, name, parameters, command):
        self.name = name
        self.parameters = {k: Parameter(k, v) for (k, v) in parameters.items()}
        self.command = command
        assert isinstance(self.command, str)

    def _validate_parameters(self, user_parameters):
        missing_params = []
        for name in self.parameters:
            if name not in user_parameters and self.parameters[name].default is None:
                missing_params.append(name)
        if len(missing_params) == 1:
            raise ExecutionException(
                "No value given for missing parameter: '%s'" % missing_params[0])
        elif len(missing_params) > 1:
            raise ExecutionException(
                "No value given for missing parameters: %s" %
                ", ".join(["'%s'" % name for name in missing_params]))

    def compute_parameters(self, user_parameters, storage_dir):
        """
        Given a dict mapping user-specified param names to values, computes parameters to
        substitute into the command for this entry point. Returns a tuple (params, extra_params)
        where `params` contains key-value pairs for parameters specified in the entry point
        definition, and `extra_params` contains key-value pairs for additional parameters passed
        by the user.

        Note that resolving parameter values can be a heavy operation, e.g. if a remote URI is
        passed for a parameter of type `path`, we download the URI to a local path within
        `storage_dir` and substitute in the local path as the parameter value.
        """
        if user_parameters is None:
            user_parameters = {}
        # Validate params before attempting to resolve parameter values
        self._validate_parameters(user_parameters)
        final_params = {}
        extra_params = {}

        for name, param_obj in self.parameters.items():
            if name in user_parameters:
                final_params[name] = param_obj.compute_value(user_parameters[name], storage_dir)
            else:
                final_params[name] = self.parameters[name].default
        for name in user_parameters:
            if name not in final_params:
                extra_params[name] = user_parameters[name]
        return _sanitize_param_dict(final_params), _sanitize_param_dict(extra_params)

    def compute_command(self, user_parameters, storage_dir):
        params, extra_params = self.compute_parameters(user_parameters, storage_dir)
        command_with_params = self.command.format(**params)
        command_arr = [command_with_params]
        command_arr.extend(["--%s %s" % (key, value) for key, value in extra_params.items()])
        return " ".join(command_arr)



class Parameter(object):
    """A parameter in an MLproject entry point."""
    def __init__(self, name, yaml_obj):
        self.name = name
        if isinstance(yaml_obj, str):
            self.type = yaml_obj
            self.default = None
        else:
            self.type = yaml_obj.get("type", "string")
            self.default = yaml_obj.get("default")

    def _compute_uri_value(self, user_param_value):
        if not data.is_uri(user_param_value):
            raise ExecutionException("Expected URI for parameter %s but got "
                                     "%s" % (self.name, user_param_value))
        return user_param_value

    def _compute_path_value(self, user_param_value, storage_dir):
        if not data.is_uri(user_param_value):
            if not os.path.exists(user_param_value):
                raise ExecutionException("Got value %s for parameter %s, but no such file or "
                                         "directory was found." % (user_param_value, self.name))
            return os.path.abspath(user_param_value)
        basename = os.path.basename(user_param_value)
        dest_path = os.path.join(storage_dir, basename)
        if dest_path != user_param_value:
            data.download_uri(uri=user_param_value, output_path=dest_path)
        return os.path.abspath(dest_path)

    def compute_value(self, user_param_value, storage_dir):
        if self.type != "path" and self.type != "uri":
            return user_param_value
        if self.type == "uri":
            return self._compute_uri_value(user_param_value)
        return self._compute_path_value(user_param_value, storage_dir)


def _sanitize_param_dict(param_dict):
    return {str(key): shlex_quote(str(value)) for key, value in param_dict.items()}

def _run_local(uri, entry_point, version, parameters, experiment_id, use_conda, use_temp_cwd,
               storage_dir):
    """
    Run an MLflow project from the given URI in a new directory.

    Supports downloading projects from Git URIs with a specified version, or copying them from
    the file system. For Git-based projects, a commit can be specified as the `version`.
    """

    # Get the working directory to use for running the project & download it there
#    work_dir = _get_work_dir(uri, use_temp_cwd)
    work_dir = "H:\\2018_git_task\\Demo_MLProject\\train"


    # Load the MLproject file
    spec_file = os.path.join(work_dir, "MLproject")
    if not os.path.isfile(spec_file):
        raise ExecutionException("No MLproject file found in %s" % uri)
    project = Project(uri, yaml.safe_load(open(spec_file).read()))
    print(project,entry_point,work_dir,parameters,use_conda,storage_dir,experiment_id)
    _run_project(project, entry_point, work_dir, parameters, use_conda, storage_dir, experiment_id)


def _run_project(project, entry_point, work_dir, parameters, use_conda, storage_dir, experiment_id):
    """Locally run a project that has been checked out in `work_dir`."""
    mlflow.set_tracking_uri('..\\')#added by cliicy
    if storage_dir is not None and not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    storage_dir_for_run = tempfile.mkdtemp(dir=storage_dir)
    print("=== Created directory %s for downloading remote URIs passed to arguments of "
           "type 'path' ===" % storage_dir_for_run)
    # Try to build the command first in case the user mis-specified parameters
    run_project_command = project.get_entry_point(entry_point).compute_command(parameters, storage_dir_for_run)
    commands = []

    # Create a new run and log every provided parameter into it.
    active_run = tracking.start_run(experiment_id=experiment_id,
                                    source_name=project.uri,
                                    source_version=tracking._get_git_commit(work_dir),
                                    entry_point_name=entry_point,
                                    source_type=SourceType.PROJECT)
    for key, value in parameters.items():
        active_run.log_param(Param(key, value))
    # Add the run id into a magic environment variable that the subprocess will read,
    # causing it to reuse the run.
    exp_id = experiment_id or tracking._get_experiment_id()
    env_map = {
        tracking._RUN_NAME_ENV_VAR: active_run.run_info.run_uuid,
        tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri(),
        tracking._EXPERIMENT_ID_ENV_VAR: str(exp_id),
    }

    commands.append(run_project_command)
    command = " && ".join(commands)
    print("=== Running command: %s ===" % command)
    try:
        command="python my_train.py 0.4 0.1"
        print("will run command aaaaa "+command +" "+work_dir +" aaaaa ")
        process.exec_cmd(command,cwd=work_dir,stream_output=True, env=env_map)
        #process.exec_cmd([os.environ.get("SHELL", "bash"), "-c", command], cwd=work_dir,
        #                 stream_output=True, env=env_map)
        tracking.end_run()
        print("=== Run succeeded ===")
    except process.ShellCommandException:
        tracking.end_run("FAILED")
        print("=== Run failed ===")

if __name__=="__main__":
    print("just try invoke mlflow")
    command = r"mlflow run --no-conda H:\2018_git_task\Demo_MLProject\train -P alpha=0.4"
    uri="H:\2018_git_task\Demo_MLProject\train"
    parameters={"alpha":0.5,"l1_ratio":0.1}
    _run_local(uri=uri, entry_point="main", version=None, parameters=parameters,
               experiment_id=None, use_conda=False, use_temp_cwd=True,
               storage_dir=None)
 #   os.system(command)

