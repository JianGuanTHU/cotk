'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import os
import os.path
import json
import argparse
import re

import requests
from . import main, _utils

DASHBOARD_URL = main.DASHBOARD_URL
QUERY_URL = DASHBOARD_URL + "/get?id=%d"

def get_result_from_id(query_id):
	'''Query uploaded info from id'''
	query = requests.get(QUERY_URL % query_id)
	if not query.ok:
		raise RuntimeError("Cannot fetch result from id %d" % query_id)
	else:
		return json.loads(query.text)

def clone_codes_from_commit(git_user, git_repo, git_commit):
	'''Download codes from commit'''
	_utils.git_clone(git_user, git_repo)
	os.chdir(git_repo)
	_utils.git_checkout_commit(git_commit)
	code_dir = os.getcwd()
	os.chdir("..")
	return code_dir

def download(args):
	'''Entrance of download'''
	parser = argparse.ArgumentParser(prog="cotk download", \
		description='Download model and information from github or dashboard.',\
		formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("model", type=str, help=\
r"""A string indicates model path. It can be one of following:
* Id of cotk dashboard. Example: 12.
* Url of a git repo / branch / commit.
    Example: https://github.com/USER/REPO
             https://github.com/USER/REPO/tree/BRANCH
             https://github.com/USER/REPO/commit/COMMIT (full commit id should be included)
* A string specifying a git repo / branch / commit.
    Example: USER/REPO
             USER/REPO/BRANCH
             USER/REPO/COMMIT (full commit id should be included)
""")
	parser.add_argument("--result", type=str, default=None, \
						help="Only works when model is from dashboard. \
Path to dump dashboard result.")
	cargs = parser.parse_args(args)

	if cargs.model.isdigit():
		# download from dashboard
		board_id = int(cargs.model)
		main.LOGGER.info("Collecting info from id %d...", board_id)
		info = get_result_from_id(board_id)
		if cargs.result is not None:
			json.dump(info, open(cargs.result, "w"))
			main.LOGGER.info("Info from id %d saved to %s.", board_id, cargs.result)

		code_dir = clone_codes_from_commit(info['git_user'], info['git_repo'], \
												  info['git_commit'])
		main.LOGGER.info("Codes from id %d fetched.")
	else:
		# download from online git repo
		patterns_2 = r'(?:https?://github\.com/)?([^\s/]+)/([^\s/]+)/?'
		patterns_3 = r'(?:https?://github\.com/)?([^\s/]+)/([^\s/]+)/(?:(?:tree|commit)/)?([^\s/]+)/?'
		match_res = re.fullmatch(patterns_2, cargs.model)
		if match_res:
			git_user, git_repo = match_res.groups()
			git_commit = "master"
		else:
			match_res = re.fullmatch(patterns_3, cargs.model)
			if not match_res:
				raise ValueError("'%s' can't match any pattern." % cargs.model)
			git_user, git_repo, git_commit = match_res.groups()

		main.LOGGER.info("Fetching {}/{}/{}".format(git_user, git_repo, git_commit))
		code_dir = clone_codes_from_commit(git_user, git_repo, git_commit)
		main.LOGGER.info("Codes from {}/{}/{} fetched.".format(git_user, git_repo, git_commit))

		config_path = "{}/.model_config.json".format(code_dir)
		if not os.path.isfile(config_path):
			raise FileNotFoundError("Config file ({}) is not found.".format(config_path))
		try:
			info = json.load(open(config_path, "r"))
		except json.JSONDecodeError as err:
			raise json.JSONDecodeError("{} is not a valid json. {}".format(config_path, err.msg), \
										err.doc, err.pos)

		if 'args' not in info:
			info['args'] = []
		if 'working_dir' not in info or 'working_dir' == '':
			info['working_dir'] = "."
		if 'entry' not in info:
			info['entry'] = "main"

	# cmd construction
	cmd = "cd {}/{} && cotk run --entry {} --only-run".\
			format(code_dir, info['working_dir'], info['entry'])
	if not isinstance(info['args'], list):
		raise ValueError("`args` in `config.json` should be of type `list`.")

	cmd += " {}".format(" ".join(info['args']))
	with open("run_model.sh", "w") as file:
		file.write(cmd)
	main.LOGGER.info("Model running cmd written in {}".format("run_model.sh"))
	print("Model running cmd: \t{}".format(cmd))

	# run model
	# result_path = "{}/result.json".format(code_dir)
	# old_time_stamp = 0
	# if os.path.exists(result_path):
	# 	old_time_stamp = os.path.getmtime(result_path)

	# os.system("bash {}/run_model.sh".format(extract_dir))
	# if not os.path.exists(result_path) or \
	# 	os.path.getmtime('{}/result.json'.format(code_dir)) <= old_time_stamp:
	# 	raise FileNotFoundError("New result file not found.")
	# print(json.load(open("{}/result.json".format(code_dir), "r")))
