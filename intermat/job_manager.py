import glob, os
from jarvis.db.jsonutils import loadjson, dumpjson

mem = []
sub_json_file = str(os.path.join(os.path.dirname(__file__), "submitted.json"))


def get_json_data():
    mem = loadjson(sub_json_file)
    return mem


def get_submitted_jobs(path="/working/knc6/InterfacesMr/Int*/submit_job"):
    mem = []
    for i in glob.glob(path):   
     if '*' not in i:
        tmp = i.split("/")[-2]
        mem.append(tmp)
    print("path,mem", path, len(mem))
    return mem


def compile_jobs():
    jobs = get_json_data()
    ini = len(jobs)
    paths = [
        "/working/knc6/InterfacesMr/Int*/submit_job",
        "/rk3/knc6/InterfacesMrDobby/Int*/submit_job",
    ]
    for i in paths:
        tmp = get_submitted_jobs(i)
        if tmp not in jobs:
            jobs.append(i)
    fin = len(jobs)
    print("jobs added", fin - ini, fin, ini)

    return jobs


if __name__ == "__main__":
    x = compile_jobs()
